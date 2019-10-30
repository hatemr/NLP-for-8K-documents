# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import matplotlib.pyplot as plt

#%%
pickle_in = open("models/lda.pickle","rb")
model_lda = pickle.load(pickle_in)
pickle_in.close()


pickle_in = open("models/svd.pickle","rb")
model_svd = pickle.load(pickle_in)
pickle_in.close()

#%%
lda = model_lda['lda']
svd = model_svd['svd']

#%%
print(lda.best_score_)
print(svd.best_score_)


d = {'DR_method': ['svd', 'lda'], 'CV_AUC': [lda.best_score_, svd.best_score_]}
df = pd.DataFrame(data=d)

ax = df.plot.bar(x='DR_method', y='CV_AUC', rot=0)
#%%
plt.xlabel('DR_method')
plt.ylabel('CV_AUC')
plt.bar(df.DR_method, df.CV_AUC, width=0.5);

for i, v in enumerate(df.CV_AUC.values):
    ax.text(v + .0, i + .0, str(v), color='red', fontweight='bold')