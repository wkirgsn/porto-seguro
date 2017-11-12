import numpy as np
from os.path import join

import multiprocessing
import uuid


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    import pandas as pd
    from sklearn import metrics
    from tpot import TPOTClassifier

    from kirgsn import reducing

    path_input_extended = join('input', 'extended')

    tpot = TPOTClassifier(generations=10, population_size=40, verbosity=2,
                          scoring='roc_auc', cv=5, #max_time_mins=60*3,
                          random_state=1990, n_jobs=-1,
                          periodic_checkpoint_folder='out')
    train = pd.read_csv(join(path_input_extended, 'train.csv'), na_values="-1")
    test = pd.read_csv(join(path_input_extended, 'test.csv'), na_values="-1")

    cols = [c for c in train.columns if c not in ['id', 'target']]
    tpot.fit(train[cols], train['target'])
    tpot.export('out/tpotted.py')
    test['target'] = tpot.predict_proba(test[cols])[:, 1]
    test[['id', 'target']].to_csv('out/submissions/tpot_{}_{}.csv.gz'.format(
        str(uuid.uuid4()).split(sep='-')[0]),
        index=False,
        float_format='%.5f',
        compression='gzip')
