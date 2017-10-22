"""
Author: WKirgsn, 2017
"""
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import gc
import uuid
from os.path import join
from itertools import accumulate, combinations

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_array
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

from kirgsn import reducing
from kirgsn import feature_engineering

SEED = 2017
TESTSIZE = 0.20
TESTSPLITS = int(1 / TESTSIZE)
N_COMP_PREPROCESSING = 10
DEBUG_FLAG = False
DECOMP_FLAG = True
ACCUMULATE_BINS_FLAG = True

cat_folder = 'cat_training'
path_feature_importances = join('out', 'feature_importance')
path_models = join('out', 'models')
path_submissions = join('out', 'submissions')
script_run_id = str(uuid.uuid4()).split('-')[0]

# catboost parameters
params_cat = {'iterations': 1500,
              'learning_rate': 0.04,
              'depth': 6,
              'loss_function': 'Logloss',
              'eval_metric': 'AUC',
              # 'eval_metric': GiniMetric(),
              'random_seed': SEED,
              'od_type': 'Iter',
              'od_wait': 50,
              'l2_leaf_reg': 4.7,
              'use_best_model': True,
              'train_dir': cat_folder,
              }

# combinations for multiplied features
float_combs = [('ps_car_13', 'ps_reg_03'),  # magic
                 # following had high pearson corr
                 ('ps_reg_01', 'ps_reg_02'),
                 ('ps_reg_01', 'ps_reg_03'),
                 ('ps_reg_02', 'ps_reg_03'),
                 ('ps_car_12', 'ps_car_13'),
                 ('ps_car_13', 'ps_car_15'),
               ]


class GiniMetric(object):

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, predictions, labels, weight):

        assert len(predictions) == 1
        assert len(labels) == len(predictions[0])

        preds = predictions[0]

        weight_sum = float(len(preds))
        error_sum = auc_gini(labels, preds)

        return error_sum, weight_sum


def auc_gini(a, p):
    fpr, tpr, thr = metrics.roc_curve(a, p, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) - 1
    return g


def ngini(a, p):
    # basic gini
    def gini(actual, pred):
        n = len(actual)
        a_s = actual[np.argsort(pred)]
        a_c = a_s.cumsum()
        giniSum = a_c.sum() / a_c[-1] - (n + 1) / 2.0
        return giniSum / n

    if p.ndim == 2:  # Required for sklearn wrapper
        p = p[:, 1]
    # normalized gini
    return gini(a, p) / gini(a, a)


def print_gini(actual, pred):
    print('gini on validation set:', ngini(actual, pred))


# XGBoost
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = ngini(labels, preds)
    return [('gini', gini_score)]


# LightGBM
def gini_lgb(actuals, preds):
    return 'gini', ngini(actuals, preds), True


# SKlearn
gini_sklearn = metrics.make_scorer(ngini, True, True)


def one_hot_encode(_train, _test, _cols):
    OHenc = OneHotEncoder()
    label_fitting_frame = \
        _train[_cols].append(_test[_cols])
    OHenc.fit(label_fitting_frame)
    del label_fitting_frame
    OH_cols = ['OH_{}_{}'.format(i, c) for i, c in enumerate(
        list(OHenc.active_features_))]

    OH_train = OHenc.transform(_train.loc[:, _cols])
    OH_test = OHenc.transform(_test.loc[:, _cols])

    encoded_train_frame = pd.DataFrame(OH_train.toarray(),
                                       columns=OH_cols,
                                       dtype=np.uint8)
    encoded_test_frame = pd.DataFrame(OH_test.toarray(),
                                      columns=OH_cols,
                                      dtype=np.uint8)
    # todo: Dont drop them for decision tree models! (this induces more
    # weight on categorical features though)
    #_train.drop(_cols, axis=1, inplace=True)
    #_test.drop(_cols, axis=1, inplace=True)

    _train = pd.concat([encoded_train_frame, _train], axis=1)
    _test = pd.concat([encoded_test_frame, _test], axis=1)
    del OHenc, OH_train, OH_test
    del encoded_test_frame, encoded_train_frame; gc.collect()
    return _train, _test


def apply_parallel(df_groups, _func):
    nthreads = multiprocessing.cpu_count()  # >> 1
    print("nthreads: {}".format(nthreads))

    res = Parallel(n_jobs=nthreads)(delayed(_func)(grp.copy()) for _, grp
                                    in df_groups)
    return pd.concat(res)


def get_decomp_feats(n_comp, _train, _test, rand_state):
    """Get decomposition features from given train and test.

    :param n_comp: number components to produce.
    :param _train: training dataframe
    :param _test: test dataframe
    :param rand_state: random State
    :return: decomposed featuers as df
    """
    print('get decomp features...')
    decomp_dict = {'tsvd': TruncatedSVD,
                   'pca': PCA,
                   'ica': FastICA,
                   'grp': GaussianRandomProjection,
                   'srp': SparseRandomProjection,
                   }
    decomp_params = {'n_components': n_comp,
                     'random_state': rand_state,
                     }
    train_decomps = pd.DataFrame()
    test_decomps = pd.DataFrame()

    for lbl, decomp_func in decomp_dict.items():
        if lbl == 'grp':
            mdl = decomp_func(eps=0.1, **decomp_params)
        elif lbl == 'srp':
            mdl = decomp_func(dense_output=True, **decomp_params)
        else:
            mdl = decomp_func(**decomp_params)
        res_train = mdl.fit_transform(_train)
        res_test = mdl.transform(_test)
        for i in range(1, n_comp + 1):
            train_decomps[lbl+'_{}'.format(i)] = res_train[:, i -1].tolist()
            test_decomps[lbl+'_{}'.format(i)] = res_test[:, i - 1].tolist()

    gc.collect()
    return train_decomps, test_decomps


if __name__ == '__main__':
    train = pd.read_csv('input/train.csv', na_values="-1")
    test = pd.read_csv('input/test.csv', na_values="-1")

    # debug
    if DEBUG_FLAG:
        train = train.iloc[:100, :]
        test = test.iloc[:100, :]

    feat_engi = feature_engineering.FeatureEngineer(train, test)
    feat_engi.clean_data()

    print('Clean data..')
    # drop calc features (seem useless)
    calc_cols = [c for c in train.columns if '_calc_' in c]
    train.drop(calc_cols, axis=1, inplace=True)
    test.drop(calc_cols, axis=1, inplace=True)

    # store column lists
    categorical_cols = [c for c in train.columns if c.endswith('cat')]
    binary_cols = [c for c in train.columns if c.endswith('bin')]
    cols_to_use = [c for c in train.columns if c not in ['id', 'target']]
    floating_cols = [c for c in cols_to_use if c not in
                     (binary_cols+categorical_cols)]

    # amount of NaNs
    train['NaN_amount'] = train[cols_to_use].isnull().sum(axis=1)
    test['NaN_amount'] = test[cols_to_use].isnull().sum(axis=1)

    # invent ps_ind_19_bin for tracking where 16,17,18 are all zeros
    train['ps_ind_19_bin'] = ~train[['ps_ind_16_bin', 'ps_ind_17_bin',
                                     'ps_ind_18_bin']].sum(axis=1)
    test['ps_ind_19_bin'] = ~test[['ps_ind_16_bin', 'ps_ind_17_bin',
                                   'ps_ind_18_bin']].sum(axis=1)
    assert train[['ps_ind_'+str(s)+'_bin' for s in (16, 17, 18, 19)]].sum(
        axis=1).unique().shape[0] == 1, 'snap!'
    # ps_ind_ 6-9 _bin and _16-18+19_bin are already one hot encoded!
    already_oh_encoded = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
                          'ps_ind_09_bin', 'ps_ind_16_bin', 'ps_ind_17_bin',
                          'ps_ind_18_bin']
    [binary_cols.remove(a) for a in already_oh_encoded]

    # fill NaNs
    for col in categorical_cols:
        # todo: Evaluate: New category or mode[0] ?
        train[col].fillna(value=train[col].mode()[0], inplace=True)
        test[col].fillna(value=test[col].mode()[0], inplace=True)
    for col in binary_cols:
        train[col].fillna(value=train[col].mode()[0], inplace=True)
        test[col].fillna(value=test[col].mode()[0], inplace=True)
    for col in floating_cols:
        train[col].fillna(value=train[col].mean(), inplace=True)
        test[col].fillna(value=test[col].mean(), inplace=True)

    print('Engineer features...')
    # cumulative sum of binaries
    train['sum_of_all_bins'] = train[binary_cols].sum(axis=1)
    test['sum_of_all_bins'] = test[binary_cols].sum(axis=1)

    # accumulate only if it doesnt puff out ram
    if len(binary_cols) < 5 and ACCUMULATE_BINS_FLAG:
        print('accumulate', ', '.join(binary_cols))
        for j in range(2, len(binary_cols)):
            comb_list = list(combinations(binary_cols, j))
            for c in comb_list:
                train['+'.join(c)] = train[list(c)].sum(axis=1)
                test['+'.join(c)] = test[list(c)].sum(axis=1)

    # combine floating point features excessively
    for c1, c2 in float_combs:
        lbl = c1+'x'+c2
        train[lbl] = train[c1]*train[c2]
        test[lbl] = test[c1]*test[c2]
        floating_cols.append(lbl)

    # one hot encode
    train, test = one_hot_encode(train, test, categorical_cols)
    cols_to_use = [c for c in train.columns if c not in ['id', 'target']]

    # mean range (magic feature no. 2)
    d_median = train.median(axis=0)
    d_mean = train.mean(axis=0)
    for c in floating_cols:
        lbl_med = c + '_exceeds_median'
        lbl_mean = c + '_exceeds_mean'
        train[lbl_med] = (train[c].values > d_median[c]).astype(int)
        test[lbl_med] = (test[c].values > d_median[c]).astype(int)
        train[lbl_mean] = (train[c].values > d_mean[c]).astype(int)
        test[lbl_mean] = (test[c].values > d_mean[c]).astype(int)
        binary_cols.append(lbl_med)
        binary_cols.append(lbl_mean)
    del d_median, d_mean; gc.collect()

    # drop single-cardinality cols
    for x in train.columns:
        cardinality = len(train[x].unique())
        if cardinality == 1:
            train.drop(x, axis=1, inplace=True)
            test.drop(x, axis=1, inplace=True)
            print('drop column: ', x)

    print('start reducing... round 1')
    reducer = reducing.Reducer()
    train = reducer.reduce(train, verbose=False)
    test = reducer.reduce(test, verbose=False)

    if DECOMP_FLAG:
        train_decomp, test_decomp = get_decomp_feats(N_COMP_PREPROCESSING,
                                          train[floating_cols],
                                          test[floating_cols],
                                          SEED)
        train = pd.concat([train, train_decomp], axis=1)
        test = pd.concat([test, test_decomp], axis=1)

        print('start reducing... round 2')
        train = reducer.reduce(train, verbose=False)
        test = reducer.reduce(test, verbose=False)

        del train_decomp, test_decomp; gc.collect()

    assert train.shape[1] == len(list(set(train.columns))), \
        'duplicate cols present'

    expanded_cols = [c for c in train.columns if c not in ['id', 'target']]

    skf = StratifiedKFold(n_splits=TESTSPLITS, random_state=SEED)
    gini_sklearn_metric = metrics.make_scorer(ngini, True, True)

    cat_preds = pd.Series(np.zeros(test.shape[0]))
    cat_feat_importances = np.zeros(len(expanded_cols))
    num_splits = skf.get_n_splits()
    for i, (train_idx, valid_idx) in enumerate(skf.split(train[expanded_cols],
                                                train['target']), start=1):
        print('train cat', i, 'of', num_splits)
        x1 = train.ix[train_idx, expanded_cols]
        x2 = train.ix[valid_idx, expanded_cols]
        y1 = train.ix[train_idx, 'target']
        y2 = train.ix[valid_idx, 'target']
        cat = CatBoostClassifier(**params_cat)
        cat.fit(x1[expanded_cols], y1, eval_set=(x2[expanded_cols], y2),
                use_best_model=True, verbose=True)

        print('cat performance on validation set:')
        print_gini(cat.predict_proba(x2[expanded_cols])[:, 1], y2)

        cat_preds += cat.predict_proba(test[expanded_cols])[:, 1] / num_splits
        cat_feat_importances += np.asarray(cat.feature_importances_)/num_splits


    cat_weight = 1
    pipeline_weight = 0.0
    test['target'] = cat_preds*cat_weight  # +results*pipeline_weight

    test[['id', 'target']].to_csv(join(path_submissions,
                                       'sub_cat_4fold_{}_pipe_{}_{}.csv.gz'.format(
                                                                    cat_weight,
                                                                    pipeline_weight,
                                                                    script_run_id)),
                                        index=False,
                                        float_format='%.5f',
                                        compression='gzip')

    pd.Series(cat_feat_importances,
              index=expanded_cols).to_csv(join(path_feature_importances,
                                               'feat_importances_{}.csv'.format(
                                                   script_run_id)))

"""
Boruta package recommends:
Only these features are not noise:
ps_ind_01
ps_ind_03
ps_ind_05_cat
ps_ind_07_bin
ps_ind_15
ps_ind_16_bin (is OH-encoded from ind_16-18_bin)
ps_reg_01
ps_reg_02
ps_reg_03
ps_car_01_cat
ps_car_03_cat
ps_car_07_cat
ps_car_12
ps_car_13
ps_car_14
ps_car_15
"""

# todo: get parallel df transformation script from forza baseline
# todo: build FeatureEngineerClass and let him select like below:
# https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/36390
