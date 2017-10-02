"""
Author: WKirgsn, 2017
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
from data_reducing import Reducer
from catboost import CatBoostClassifier

SEED = 2017
TESTSIZE = 0.25


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert(len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))],
                     dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1*all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

reducer = Reducer()
train = reducer.reduce(train, verbose=False)
test = reducer.reduce(test, verbose=False)
col = [c for c in train.columns if c not in ['id', 'target']]

"""x1, x2, y1, y2 = model_selection.train_test_split(train[col],
                                                  train['target'],
                                                  test_size=TESTSIZE,
                                                  random_state=SEED)
"""
cat = CatBoostClassifier()
cat.fit(train[col], train['target'])

test['target'] = cat.predict(test[col])
test[['id', 'target']].to_csv('out/sub.csv.gz',
                              index=False,
                              float_format='%.5g',
                              compression='gzip')
