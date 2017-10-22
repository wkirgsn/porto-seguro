import pandas as pd
import numpy as np
from kirgsn import reducing


class FeatureEngineer(object):

    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

        conversion_dict = \
            {'int': [np.int8, np.int16, np.int32, np.int64],
             'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
             'float': [np.float32, ]}

        self.reducer = reducing.Reducer(conversion_dict)

    def clean_data(self):
        print('Clean data..')
        # drop calc features (seem useless)
        calc_cols = [c for c in self.train_df.columns if '_calc_' in c]
        self.train_df.drop(calc_cols, axis=1, inplace=True)
        self.test_df.drop(calc_cols, axis=1, inplace=True)

    def reduce_mem_usage(self):
        self.reducer.reduce(self.train_df)
        self.reducer.reduce(self.test_df)
