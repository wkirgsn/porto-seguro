"""
Author: Kirgsn, 2017
"""
import pandas as pd
import numpy as np
from itertools import combinations, compress
import gc
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier as RFC

from kirgsn import reducing


class Engineer:
    def __init__(self, train_df, test_df, seed=None):
        self.train = train_df
        self.test = test_df

        conversion_dict = \
            {'int': [np.int8, np.int16, np.int32, np.int64],
             'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
             'float': [np.float32, ]}

        self.reducer = reducing.Reducer(conversion_dict)

        self.categorical_cols = []
        self.binary_cols, self.oh_cols = [], []
        self.floating_cols = []
        self.cols_to_use = []
        self.update_col_lists()

        self.rand_state = seed
        # combinations for multiplied float features
        self.float_combs = [('ps_car_13', 'ps_reg_03'),  # magic
                            # following had high pearson corr
                            ('ps_reg_01', 'ps_reg_02'),
                            ('ps_reg_01', 'ps_reg_03'),
                            ('ps_reg_02', 'ps_reg_03'),
                            ('ps_car_12', 'ps_car_13'),
                            ('ps_car_13', 'ps_car_15'),
                            ]

        self.decomp_dict = {'tsvd': TruncatedSVD,
                            'pca': PCA,
                            'ica': FastICA,
                            'grp': GaussianRandomProjection,
                            'srp': SparseRandomProjection,
                            }
        self.agenda = []

    def update_col_lists(self):
        self.categorical_cols = \
            [c for c in self.train.columns if c.endswith('cat')]
        self.binary_cols = [c for c in self.train.columns if c.endswith('bin')]
        self.oh_cols = [c for c in self.train.columns if c.endswith('_oh')]
        self.cols_to_use = \
            [c for c in self.train.columns if c not in ['id', 'target']]
        self.floating_cols = \
            [c for c in self.cols_to_use if c not in
             (self.binary_cols + self.categorical_cols + self.oh_cols)]

    def add_agenda(self, todolist):
        self.agenda = todolist

    def process_the_agenda(self):
        assert len(self.agenda) > 0, 'Engineer has no agenda to work off'
        for item, kwargs in self.agenda:
            self.item(**kwargs)


class FeatureEngineer(Engineer):

    def clean_data(self):
        print('Clean data..')
        # drop calc features (seem useless)
        calc_cols = [c for c in self.train.columns if '_calc_' in c]
        if len(calc_cols) > 0:
            self.train.drop(calc_cols, axis=1, inplace=True)
            self.test.drop(calc_cols, axis=1, inplace=True)
            self.update_col_lists()

        # drop single-cardinality cols
        cardinalities = (len(self.train[x].unique()) for x in self.cols_to_use)
        single_card_cols = compress(self.cols_to_use,
                                    (card == 1 for card in cardinalities))
        for frame in (self.train, self.test):
            frame.drop(single_card_cols, axis=1, inplace=True)
        [print('drop column: ', x) for x in single_card_cols]
        self.update_col_lists()

    def reduce_mem_usage(self):
        self.train = self.reducer.reduce(self.train)
        self.test = self.reducer.reduce(self.test)

    def add_nan_per_row(self):
        """add amount of NaNs per row as new feature"""
        self.train['NaN_amount'] = \
            self.train[self.cols_to_use].isnull().sum(axis=1)
        self.test['NaN_amount'] = \
            self.test[self.cols_to_use].isnull().sum(axis=1)

    def add_ind_19(self):
        """Invent ps_ind_19_bin for tracking where 16,17,18 are all zeros.

        ps_ind_ 6-9 _bin and _16-18+19_bin are already one hot encoded!
        """
        already_oh_encoded = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
                              'ps_ind_09_bin', 'ps_ind_16_bin', 'ps_ind_17_bin',
                              'ps_ind_18_bin',
                              'ps_ind_19_bin'  # to add in the following
                              ]
        for frame in (self.train, self.test):
            frame['ps_ind_19_bin'] = 1 - frame[['ps_ind_16_bin',
                                                  'ps_ind_17_bin',
                                                  'ps_ind_18_bin']].sum(axis=1)
            assert frame[['ps_ind_' + str(s) + '_bin'
                           for s in
                           (16, 17,
                            18, 19)]].sum(axis=1).unique().shape[0] == 1,\
                'snap!'

            # rename these pre-one-hot-encoded features
            frame.rename(columns={c: c.replace('_bin', '_oh') for c in
                                  already_oh_encoded}, inplace=True)
        self.update_col_lists()

    def fillna(self):
        # fill NaNs
        for frame in (self.train, self.test):
            for c in self.categorical_cols:
                # add new category
                # Variable ps_car_03_cat has 148367 records (68.39%) with NaNs
                # Variable ps_car_05_cat has 96026 records (44.26%) with NaNs
                frame[c].fillna(value=frame[c].max()+1, inplace=True)
            for c in self.binary_cols:
                frame[c].fillna(value=frame[c].mode()[0], inplace=True)
            for c in self.floating_cols:
                frame[c].fillna(value=frame[c].mean(), inplace=True)

    def _get_sum(self, df, col_tuple):
        return df[list(col_tuple)].sum(axis=1)

    def accum_bins(self):
        # cumulative sum of binaries

        self.train['sum_of_all_bins'] = self.train[self.binary_cols].sum(axis=1)
        self.test['sum_of_all_bins'] = self.test[self.binary_cols].sum(axis=1)

        # accumulate only if it doesnt puff out ram
        if len(self.binary_cols) < 5:
            print('accumulate', ', '.join(self.binary_cols))
            with Parallel(n_jobs=cpu_count()) as prll:
                for j in range(2, len(self.binary_cols)):
                    comb_list = list(combinations(self.binary_cols, j))
                    # parallelization somehow not effective here
                    for frame in (self.test, self.train):
                        ret_list = prll(delayed(self._get_sum)(frame, c_pair)
                                        for c_pair in comb_list)

                        frame[['+'.join(c) for c in comb_list]] = \
                            pd.concat(ret_list, axis=1)
        else:
            print('dont accumulate bins as there are too many:',
                  len(self.binary_cols))
            [print(c) for c in self.binary_cols]
        # todo: change cols to end with 'accum'
        self.update_col_lists()

    def combine_float_features(self):
        # add interaction variables
        for frame in (self.train, self.test):
            poly = PolynomialFeatures(degree=2, interaction_only=False,
                                      include_bias=False)
            poly_feats_obj = poly.fit_transform(frame[self.floating_cols])
            inter_cols = [s.replace(' ', 'x') for s in
                          poly.get_feature_names(self.floating_cols)]

            interactions_df = pd.DataFrame(data=poly_feats_obj,
                                           columns=inter_cols)

            frame[interactions_df.columns] = \
                pd.concat([interactions_df[c] for c in
                           interactions_df.columns], axis=1)

        self.update_col_lists()

        # arithmetic shenanigans
        for c in self.floating_cols:
            for df in (self.train, self.test):
                df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
                df[c+str('_exp')] = np.exp(df[c].values) - 1
        self.update_col_lists()

    def one_hot_encode(self):

        print('one-hot encode', ', '.join(self.categorical_cols))
        label_fitting_frame = pd.concat([self.train[self.categorical_cols],
                                        self.test[self.categorical_cols]],
                                        axis=0)

        dums = pd.get_dummies(data=label_fitting_frame,
                              prefix=[s.replace('_cat', '') for s in
                                      self.categorical_cols],
                              columns=label_fitting_frame.columns)
        delim = self.train.index.max() + 1
        self.train = pd.concat([self.train, dums.iloc[:delim, :]], axis=1)
        self.test = pd.concat([self.test, dums.iloc[delim:, :]], axis=1)

        for frame in (self.train, self.test):
            frame.rename({orig: orig+'_oh' for orig in dums.columns},
                         inplace=True)
            frame.drop(self.categorical_cols, axis=1, inplace=True)

        del dums
        gc.collect()
        self.update_col_lists()

    def add_decomp_feats(self, n_comp):

        decomp_params = {'n_components': n_comp,
                         'random_state': self.rand_state,
                         }

        for lbl, decomp_func in self.decomp_dict.items():
            if lbl == 'grp':
                mdl = decomp_func(eps=0.1, **decomp_params)
            elif lbl == 'srp':
                mdl = decomp_func(dense_output=True, **decomp_params)
            else:
                mdl = decomp_func(**decomp_params)
            res_train = mdl.fit_transform(self.train[self.floating_cols])
            res_test = mdl.transform(self.test[self.floating_cols])
            for i in range(1, n_comp + 1):
                self.train[lbl+'_{}'.format(i)] = res_train[:, i - 1].tolist()
                self.test[lbl+'_{}'.format(i)] = res_test[:, i - 1].tolist()

        gc.collect()
        self.update_col_lists()
        assert self.train.shape[1] == len(list(set(self.train.columns))), \
            'duplicate cols present'

    def transform_high_cardinalities(self):
        """Suggestions on transformations:
        https://www.kdnuggets.com/2016/08/include-high-cardinality-attributes-predictive-model.html
        """
        # ps_car_11_cat has 104 distinct values
        def _woe(s, tp, tn):
            """Weight of evidence

            woe_i = ln(P_i/TP) - ln(N_i/TN)

            :param s: pandas groupby obj
            :param tp: total positives in full series (target prior)
            :param tn: total negatives in full series
            """
            p = s.sum()
            nom = p / tp
            den = (s.count() - p) / tn
            return np.log(nom/den)

        def _micci_barreca_encode(s, tp, min_samples_leaf=1, smoothing=1):
            """Source:
            https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
            """
            # this is something between supervised ratio and target prior
            smoothing = \
                1 / (1 + np.exp(-(s.count() - min_samples_leaf) / smoothing))
            return tp * (1-smoothing) + s.mean() * smoothing

        def add_noise(df, noise_level):
            col_filter = [col for col in df.columns if '_cat' not in col]
            tmp = df.loc[:, col_filter]
            df.loc[:, col_filter] = \
                tmp * (1 + noise_level * np.random.randn(*tmp.shape))
            return df

        card_thresh = 20
        noise_rate = .01
        target_prior = self.train['target'].sum()
        target_size = self.train['target'].count()
        aggregation_agenda = \
            {'_woe': lambda x: _woe(x, target_prior, target_size-target_prior),
             '_micci': lambda x: _micci_barreca_encode(x, target_prior,
                                                       min_samples_leaf=100,
                                                       smoothing=10),
             }

        cardinalities = (len(self.train[x].unique()) for x in
                         self.categorical_cols)
        high_card_cols = list(compress(self.categorical_cols,
                                  (card > card_thresh for card in
                                   cardinalities)))

        print('transform high cardinality features: ', ', '.join(high_card_cols))
        for col in high_card_cols:
            # transform to other representations
            cat_perc = \
                self.train[[col, 'target']]\
                    .groupby([col], as_index=False).target\
                    .agg(aggregation_agenda)\
                    .rename(columns={agg_key: col.replace('_cat', agg_key) for
                                     agg_key in aggregation_agenda.keys()})

            # add noise to mitigate over-fitting
            cat_perc = add_noise(cat_perc, noise_level=noise_rate)

            # patch new floating features to original df
            self.train = self.train.merge(cat_perc, how='inner', on=col)
            self.test = self.test.merge(cat_perc, how='inner', on=col)

        for frame in (self.train, self.test):
            frame.drop(high_card_cols, axis=1, inplace=True)
        self.update_col_lists()


class BaggingEngineer(Engineer):
    """This class is probably obsolete due to the existence of
    sklearn.feature_selection.SelectFromModel"""
    class Bag:
        pass

    def __init__(self, *args):
        super().__init__(*args)
        self.top_feats = []
        self.weights = np.empty(len(self.cols_to_use))
        self.weights.fill(1/len(self.cols_to_use))

    def bag_around(self, model=None):
        """Source:
        https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/36390
        """
        print('bag around..')
        n_rows = self.train.shape[0]

        best_feats = []
        best_score = 0
        epochs = 3000
        bag_size = 30  # todo: optimize
        w_growth_rate = .2
        w_thresh = .2
        kfold_splits = 5

        skf = StratifiedKFold(n_splits=kfold_splits,
                              random_state=self.rand_state)
        scores = []

        for epoch in range(epochs):
            feat_bag_ids = np.random.choice(a=len(self.cols_to_use),
                                            size=bag_size,
                                            replace=False,
                                            p=self.weights)
            feats_in_bag = self.top_feats + [s for s in compress(
                self.cols_to_use, feat_bag_ids)]
            prediction_val = np.zeros(n_rows)
            for i, (train_idx, valid_idx) in enumerate(
                    skf.split(self.train[feats_in_bag],  # columns w/out effect
                              self.train['target']), start=1):
                x1 = self.train.loc[train_idx, feats_in_bag]
                x2 = self.train.loc[valid_idx, feats_in_bag]
                y1 = self.train.loc[train_idx, 'target']
                y2 = self.train.loc[valid_idx, 'target']
                model = RFC(100, max_depth=12, max_features=20,
                            min_samples_leaf=4,
                            n_jobs=cpu_count(),
                            random_state=self.rand_state)
                model.fit(x1, y1)
                prediction_val[valid_idx] = model.predict_proba(x2)[:, 0]
            score = roc_auc_score(self.train['target'],
                                  prediction_val)
            scores.append(score)
            if epoch < 30:
                print('epoch', epoch, '-', score)
            else:
                mean_windowed_score = np.mean(scores[-30:])
                if epoch % 25 == 0:
                    print('epoch', epoch, '- mean windowed score:',
                          mean_windowed_score, '- score', score)

                rate = (1+w_growth_rate) if score > mean_windowed_score \
                    else (1-w_growth_rate)
                self.weights[feat_bag_ids] *= rate
                if score > best_score:
                    best_score = score
                    best_feats = feats_in_bag

                self.weights /= np.sum(self.weights)
                max_weight = np.max(self.weights)

                # add feat to the top feats if weight is > threshold
                if max_weight > w_thresh:
                    new_feats = [(i, f) for i, (f, w) in enumerate(zip(
                        self.cols_to_use, list(self.weights))) if w > w_thresh]
                    # top_weights = new_feats_s.values
                    self.top_feats = self.top_feats + [f for _, f in new_feats]
                    self.weights = np.delete(self.weights,
                                             [i for i, _ in new_feats])
                    self.weights /= np.sum(self.weights)


class EnsembleEngineer(Engineer):
    # todo: implement
    pass