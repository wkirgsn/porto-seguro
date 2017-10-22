"""
Author: Kirgsn, 2017
"""
import numpy as np
import time
import gc


class Reducer:
    """
    Class that takes a dict of increasingly big numpy datatypes to transform
    the data of a pandas dataframe to in order to save memory usage.
    """
    memory_scale_factor = 1024**2  # memory in MB

    def __init__(self, conv_table=None):
        """
        :param conv_table: dict with np.dtypes-strings as keys
        """
        if conv_table is None:
            self.conversion_table = \
                {'int': [np.int8, np.int16, np.int32, np.int64],
                 'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                 'float': [np.float16, np.float32, ]}

    def _type_candidates(self, k):
        for c in self.conversion_table[k]:
            i = np.iinfo(c) if 'int' in k else np.finfo(c)
            yield c, i

    def reduce(self, df, verbose=True):
        """Takes a dataframe and returns it with all data transformed to the
        smallest necessary types.

        :param df: pandas dataframe
        :param verbose: If True, outputs more information
        :return: pandas dataframe with reduced data types
        """
        mem_usage_orig = df.memory_usage().sum() / self.memory_scale_factor
        start_time = time.time()
        for col in df.columns:
            # skip NaNs
            if df[col].isnull().any():
                if verbose:
                    print(col, 'has NaNs - Skip..')
                continue
            # detect kind of type
            coltype = df[col].dtype
            if np.issubdtype(coltype, np.integer):
                conv_key = 'int' if df[col].min() < 0 else 'uint'
            elif np.issubdtype(coltype, np.float):
                conv_key = 'float'
            else:
                if verbose:
                    print(col, 'is', coltype, '- Skip..')
                continue
            # find right candidate
            for cand, cand_info in self._type_candidates(conv_key):
                if df[col].max() <= cand_info.max and \
                                df[col].min() >= cand_info.min:
                    df[col] = df[col].astype(cand)
                    if verbose:
                        print('convert', col, 'to', str(cand))
                    break
        mem_usage_new = df.memory_usage().sum() / self.memory_scale_factor
        end_time = time.time()
        print('reduced df from {:.4} MB to {:.4} MB in {:.2} seconds'.format(
            mem_usage_orig, mem_usage_new, (end_time - start_time)))
        gc.collect()
        return df
