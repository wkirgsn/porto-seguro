import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join
"""
# Feature importance
df_feat_importances = \
    pd.Series.from_csv("out/feature_importance/feat_importances_262b5632.csv")

order = np.argsort(df_feat_importances.values)
plt.figure()
plt.title('feature importances')
plt.barh(range(df_feat_importances.shape[0]), df_feat_importances.values[order])
plt.yticks(range(df_feat_importances.shape[0]), df_feat_importances.index[
    order])
plt.show()"""
path_submissions = 'out/submissions'
df1 = pd.read_csv(join(path_submissions, 'lightgbm_sub.csv'))
df2 = pd.read_csv(join(path_submissions, 'xgb_sub.csv'))
df2.columns = [x + '_' if x not in ['id'] else x for x in df2.columns]
blend = pd.merge(df1, df2, how='left', on='id')
for c in df1.columns:
    if c != 'id':
        blend[c] = (blend[c] * 0.07) + (blend[c + '_'] * 0.03)
blend = blend[df1.columns]
blend['target'] = (np.exp(blend['target'].values) - 1.0).clip(0, 1)
blend.to_csv(join(path_submissions, 'blend_sub_{}.csv.gz'.format(
    1)), index=False, float_format='%.5f', compression='gzip')

