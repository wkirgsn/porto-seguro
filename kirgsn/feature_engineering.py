import pandas as pd
import numpy as np
from itertools import accumulate, combinations
import gc
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from sklearn.preprocessing import OneHotEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD

from kirgsn import reducing


class FeatureEngineer(object):

    def __init__(self, train_df, test_df, seed=None):
        self.train = train_df
        self.test = test_df

        conversion_dict = \
            {'int': [np.int8, np.int16, np.int32, np.int64],
             'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
             'float': [np.float32, ]}

        self.reducer = reducing.Reducer(conversion_dict)

        self.categorical_cols = []
        self.binary_cols = []
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

    def update_col_lists(self):
        self.categorical_cols = \
            [c for c in self.train.columns if c.endswith('cat')]
        self.binary_cols = [c for c in self.train.columns if c.endswith('bin')]
        self.cols_to_use = \
            [c for c in self.train.columns if c not in ['id', 'target']]
        self.floating_cols = \
            [c for c in self.cols_to_use if c not in
             (self.binary_cols + self.categorical_cols)]

    def clean_data(self):
        print('Clean data..')
        # drop calc features (seem useless)
        calc_cols = [c for c in self.train.columns if '_calc_' in c]
        if len(calc_cols) > 0:
            self.train.drop(calc_cols, axis=1, inplace=True)
            self.test.drop(calc_cols, axis=1, inplace=True)

        # drop single-cardinality cols
        for x in self.train.columns:
            cardinality = len(self.train[x].unique())
            if cardinality == 1:
                self.train.drop(x, axis=1, inplace=True)
                self.test.drop(x, axis=1, inplace=True)
                print('drop column: ', x)

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
        self.train['ps_ind_19_bin'] = 1 - \
                                      self.train[['ps_ind_16_bin',
                                                  'ps_ind_17_bin',
                                                  'ps_ind_18_bin']].sum(axis=1)
        self.test['ps_ind_19_bin'] = 1 - \
                                     self.test[['ps_ind_16_bin',
                                                'ps_ind_17_bin',
                                                'ps_ind_18_bin']].sum(axis=1)
        assert self.train[['ps_ind_' + str(s) + '_bin'
                           for s in
                           (16, 17,
                            18, 19)]].sum(axis=1).unique().shape[0] == 1,\
            'snap!'

        already_oh_encoded = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
                              'ps_ind_09_bin', 'ps_ind_16_bin', 'ps_ind_17_bin',
                              'ps_ind_18_bin']
        [self.binary_cols.remove(a) for a in already_oh_encoded]

    def fillna(self):
        # fill NaNs
        for c in self.categorical_cols:
            # todo: Evaluate: New category or mode[0] ?
            self.train[c].fillna(value=self.train[c].mode()[0], inplace=True)
            self.test[c].fillna(value=self.test[c].mode()[0], inplace=True)
        for c in self.binary_cols:
            self.train[c].fillna(value=self.train[c].mode()[0], inplace=True)
            self.test[c].fillna(value=self.test[c].mode()[0], inplace=True)
        for c in self.floating_cols:
            self.train[c].fillna(value=self.train[c].mean(), inplace=True)
            self.test[c].fillna(value=self.test[c].mean(), inplace=True)

    def _get_sum(self, df, col_tuple):
        return df[list(col_tuple)].sum(axis=1)

    def accum_bins(self, do_accumulate):
        # cumulative sum of binaries

        self.train['sum_of_all_bins'] = self.train[self.binary_cols].sum(axis=1)
        self.test['sum_of_all_bins'] = self.test[self.binary_cols].sum(axis=1)

        # accumulate only if it doesnt puff out ram
        if len(self.binary_cols) < 5 and do_accumulate:
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

    def combine_float_features(self):
        # combine floating point features excessively
        for c1, c2 in self.float_combs:
            lbl = c1 + 'x' + c2
            self.train[lbl] = self.train[c1] * self.train[c2]
            self.test[lbl] = self.test[c1] * self.test[c2]
            self.floating_cols.append(lbl)

        # value above mean/median
        d_median = self.train.median(axis=0)
        d_mean = self.train.mean(axis=0)
        for c in self.floating_cols:
            lbl_med = c + '_exceeds_median_bin'
            lbl_mean = c + '_exceeds_mean_bin'
            for frame in (self.train, self.test):
                frame[lbl_med] = (frame[c].values > d_median[c]).astype(int)
                frame[lbl_mean] = (frame[c].values > d_mean[c]).astype(int)
            self.binary_cols.append(lbl_med)
            self.binary_cols.append(lbl_mean)
        del d_median, d_mean
        gc.collect()

        # todo: more arithmetic shenanigans (log, e to the x, above harm. mean)

    def one_hot_encode(self):
        OHenc = OneHotEncoder()
        label_fitting_frame = \
            self.train[self.categorical_cols].append(
                self.test[self.categorical_cols])
        OHenc.fit(label_fitting_frame)
        del label_fitting_frame
        OH_cols = ['OH_{}_{}'.format(i, c) for i, c in enumerate(
            list(OHenc.active_features_))]

        OH_train = OHenc.transform(self.train.loc[:, self.categorical_cols])
        OH_test = OHenc.transform(self.test.loc[:, self.categorical_cols])

        encoded_train_frame = pd.DataFrame(OH_train.toarray(),
                                           columns=OH_cols,
                                           dtype=np.uint8)
        encoded_test_frame = pd.DataFrame(OH_test.toarray(),
                                          columns=OH_cols,
                                          dtype=np.uint8)

        self.train.drop(self.categorical_cols, axis=1, inplace=True)
        self.test.drop(self.categorical_cols, axis=1, inplace=True)

        self.train = pd.concat([encoded_train_frame, self.train], axis=1)
        self.test = pd.concat([encoded_test_frame, self.test], axis=1)
        del OHenc, OH_train, OH_test
        del encoded_test_frame, encoded_train_frame;
        gc.collect()

        self.cols_to_use = \
            [c for c in self.train.columns if c not in ['id', 'target']]

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
                self.train[lbl+'_{}'.format(i)] = res_train[:, i -1].tolist()
                self.test[lbl+'_{}'.format(i)] = res_test[:, i - 1].tolist()

        gc.collect()
        assert self.train.shape[1] == len(list(set(self.train.columns))), \
            'duplicate cols present'
