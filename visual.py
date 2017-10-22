import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Feature importance
df_feat_importances = \
    pd.Series.from_csv("out/feature_importance/feat_importances_262b5632.csv")

order = np.argsort(df_feat_importances.values)
plt.figure()
plt.title('feature importances')
plt.barh(range(df_feat_importances.shape[0]), df_feat_importances.values[order])
plt.yticks(range(df_feat_importances.shape[0]), df_feat_importances.index[
    order])
plt.show()
