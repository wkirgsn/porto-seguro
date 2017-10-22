import numpy as np
import datetime
import gc
import multiprocessing
import uuid


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
    # weight on categorical features though)
    _train.drop(_cols, axis=1, inplace=True)
    _test.drop(_cols, axis=1, inplace=True)

    _train = pd.concat([encoded_train_frame, _train], axis=1)
    _test = pd.concat([encoded_test_frame, _test], axis=1)
    del OHenc, OH_train, OH_test
    del encoded_test_frame, encoded_train_frame; gc.collect()
    return _train, _test


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    import pandas as pd
    from sklearn import metrics
    from tpot import TPOTClassifier
    from sklearn.preprocessing import OneHotEncoder

    from kirgsn import reducing

    tpot = TPOTClassifier(generations=10, population_size=40, verbosity=2,
                          scoring='roc_auc', cv=4, #max_time_mins=60*3,
                          random_state=1990, n_jobs=-1,
                          periodic_checkpoint_folder='out')
    train = pd.read_csv('input/train.csv', na_values="-1")
    test = pd.read_csv('input/test.csv', na_values="-1")

    categorical_cols = [c for c in train.columns if c.endswith('cat')]
    binary_cols = [c for c in train.columns if c.endswith('bin')]
    cols_to_use = [c for c in train.columns if c not in ['id', 'target']]
    floating_cols = [c for c in cols_to_use if c not in
                     (binary_cols + categorical_cols)]

    # Feature Engineering
    print('Feature Engineer..')

    # amount of NaNs
    train['NaN_amount'] = train[cols_to_use].isnull().sum(axis=1)
    test['NaN_amount'] = test[cols_to_use].isnull().sum(axis=1)

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
    train['ps_car_13xps_reg_03'] = train['ps_car_13'] * train['ps_reg_03']
    test['ps_car_13xps_reg_03'] = test['ps_car_13'] * test['ps_reg_03']
    floating_cols.append('ps_car_13xps_reg_03')
    # one hot encode
    train, test = one_hot_encode(train, test, categorical_cols)

    # drop single-cardinality cols
    for x in train.columns:
        cardinality = len(train[x].unique())
        if cardinality == 1:
            train.drop(x, axis=1, inplace=True)
            test.drop(x, axis=1, inplace=True)
            print('drop column: ', x)

    # consider one hot encoded and additional features
    cols_to_use = [c for c in train.columns if c not in ['id', 'target']]

    reducer = reducing.Reducer()
    train = reducer.reduce(train, verbose=False)
    test = reducer.reduce(test, verbose=False)

    tpot.fit(train[cols_to_use], train['target'])
    tpot.export('out/tpotted.py')
    test['target'] = tpot.predict_proba(test[cols_to_use])[:, 1]
    test[['id', 'target']].to_csv('out/tpot_{}_{}.csv.gz'.format(
        str(uuid.uuid4()).split(sep='-')[0],
        datetime.datetime.today()),
        index=False,
        float_format='%.5f',
        compression='gzip')
