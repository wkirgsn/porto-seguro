"""
Author: Kirgsn, 2017, https://www.kaggle.com/wkirgsn
"""
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split,\
    cross_val_score
import xgboost as xgb
from lightgbm import LGBMClassifier


def logodds(arr, inverse=False):
    """logit transformation"""
    if inverse:
        return 1/(1+np.exp(-arr))
    else:
        almost_zero = 1e-12
        almost_one = 1 - almost_zero
        arr[arr > almost_one] = almost_one
        arr[arr < almost_zero] = almost_zero
        return np.log(arr/(1-arr))


def the1owl_xgb(trn, tst, cols_to_use):
    xgb_params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'gamma': 9,
                  'colsample_bytree': 0.9, 'objective': 'binary:logistic',
                  'eval_metric': 'auc', 'seed': 2017, 'silent': True}
    x1, x2, y1, y2 = train_test_split(trn[cols_to_use], trn['target'],
                                      test_size=0.25,
                                      random_state=2017)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    mdl = xgb.train(xgb_params, xgb.DMatrix(x1, y1), 5000, watchlist,
                      maximize=True, verbose_eval=50, early_stopping_rounds=200)
    return mdl.predict(xgb.DMatrix(tst[cols_to_use]),
                       ntree_limit=mdl.best_ntree_limit + 45)


class EnsembleEngineer():
    # todo: implement
    pass


class VladDemidovEnsemble():
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                     random_state=2017).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]

                print("Fit %s fold %d" % (str(clf).split('(')[0], j + 1))
                clf.fit(X_train, y_train)

                y_pred = clf.predict_proba(X_holdout)[:, 1]

                S_train[test_idx, i] = logodds(y_pred)
                probs = clf.predict_proba(T)[:, 1]
                S_test_i[:, j] = logodds(probs)

            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3,
                                  scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:, 1]
        return res


class VladDemidovLGBMs():
    def __init__(self):
        pass

    def get_models(self):
        lgb_params = {}
        lgb_params['learning_rate'] = 0.02
        lgb_params['n_estimators'] = 650
        lgb_params['max_bin'] = 10
        lgb_params['subsample'] = 0.8
        lgb_params['subsample_freq'] = 10
        lgb_params['colsample_bytree'] = 0.8
        lgb_params['min_child_samples'] = 500
        lgb_params['random_state'] = 99

        lgb_params2 = {}
        lgb_params2['n_estimators'] = 1090
        lgb_params2['learning_rate'] = 0.02
        lgb_params2['colsample_bytree'] = 0.3
        lgb_params2['subsample'] = 0.7
        lgb_params2['subsample_freq'] = 2
        lgb_params2['num_leaves'] = 16
        lgb_params2['random_state'] = 99

        lgb_params3 = {}
        lgb_params3['n_estimators'] = 1100
        lgb_params3['max_depth'] = 4
        lgb_params3['learning_rate'] = 0.02
        lgb_params3['random_state'] = 99

        # incorporated one more layer of my defined lgb params
        lgb_params4 = {}
        lgb_params4['n_estimators'] = 1450
        lgb_params4['max_bin'] = 20
        lgb_params4['max_depth'] = 6
        lgb_params4['learning_rate'] = 0.25  # shrinkage_rate
        lgb_params4['boosting_type'] = 'gbdt'
        lgb_params4['objective'] = 'binary'
        lgb_params4['min_data'] = 500  # min_data_in_leaf
        lgb_params4['min_hessian'] = 0.05  # min_sum_hessian_in_leaf
        lgb_params4['verbose'] = 0

        return LGBMClassifier(**lgb_params),\
               LGBMClassifier(**lgb_params2),\
                LGBMClassifier(**lgb_params3),\
                LGBMClassifier(**lgb_params4)