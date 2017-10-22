"""
Author: WKirgsn, 2017
"""
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import gc
import uuid
from os.path import join

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_array
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn import metrics

from kirgsn import feature_engineering

SEED = 1990
TESTSIZE = 0.20
TESTSPLITS = int(1 / TESTSIZE)
N_COMP_PREPROCESSING = 10
DEBUG_FLAG = False
DECOMP_FLAG = True
ACCUMULATE_BINS_FLAG = True
LOAD_FEATS = True

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


def apply_parallel(df_groups, _func):
    nthreads = multiprocessing.cpu_count()  # >> 1
    print("nthreads: {}".format(nthreads))

    res = Parallel(n_jobs=nthreads)(delayed(_func)(grp.copy()) for _, grp
                                    in df_groups)
    return pd.concat(res)


if __name__ == '__main__':
    print('load data...')

    train = pd.read_csv('input/train.csv', na_values="-1")
    test = pd.read_csv('input/test.csv', na_values="-1")

    # debug
    if DEBUG_FLAG:
        train = train.iloc[:100, :]
        test = test.iloc[:100, :]

    # todo: enable expanded dataset loading and saving
    feat_engi = feature_engineering.FeatureEngineer(train, test, SEED)
    feat_engi.clean_data()
    feat_engi.update_col_lists()
    print('Engineer features...')
    feat_engi.add_nan_per_row()
    feat_engi.add_ind_19()
    feat_engi.fillna()
    feat_engi.accum_bins(do_accumulate=ACCUMULATE_BINS_FLAG)
    feat_engi.combine_float_features()
    feat_engi.one_hot_encode()
    feat_engi.clean_data()

    print('start reducing... round 1')
    feat_engi.reduce_mem_usage()

    if DECOMP_FLAG:
        print('add decomp features...')
        feat_engi.add_decomp_feats(N_COMP_PREPROCESSING)
        print('start reducing... round 2')
        feat_engi.reduce_mem_usage()

    
    # Train models
    feat_engi.update_col_lists()
    expanded_cols = feat_engi.cols_to_use

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

# todo: Feature baggin!
# https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/36390
