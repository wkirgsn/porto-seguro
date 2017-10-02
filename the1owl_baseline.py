"""
Author: The1Owl
https://www.kaggle.com/the1owl/forza-baseline
This script has been released under the Apache 2.0 open source license.
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
import xgboost as xgb

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
col = [c for c in train.columns if c not in ['id', 'target']]


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert(len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))],
                     dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1*all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_xgb(preds, y):
    y = y.get_label()
    return 'gini', gini(y, preds) / gini(y, y)

params = {'eta': 0.02,
          'max_depth': 4,
          'subsample': 0.9,
          'colsample_bytree': 0.9,
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'seed': 99,
          'silent': False}
x1, x2, y1, y2 = model_selection.train_test_split(train[col],
                                                  train['target'],
                                                  test_size=0.25,
                                                  random_state=99)
watchlist = [(xgb.DMatrix(x1, y1), 'train'),
             (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1),
                  5000,  watchlist, feval=gini_xgb,
                  maximize=True, verbose_eval=50,
                  early_stopping_rounds=100)
test['target'] = model.predict(xgb.DMatrix(test[col]),
                               ntree_limit=model.best_ntree_limit+45)
test[['id', 'target']].to_csv('out/submission.csv',
                              index=False,
                              float_format='%.5f')
