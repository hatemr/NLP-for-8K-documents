# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import matplotlib.pyplot as plt

#%%
pickle_in = open("models/lda_rf.pickle","rb")
model_lda = pickle.load(pickle_in)
pickle_in.close()

#%%
pickle_in = open("models/svd.pickle","rb")
model_svd = pickle.load(pickle_in)
pickle_in.close()

#%%
lda = model_lda['lda_rf']
svd = model_svd['svd']

#%%
print(lda.best_score_)
print(svd.best_score_)


d = {'DR_method': ['svd', 'lda'], 'CV_AUC': [lda.best_score_, svd.best_score_]}
df = pd.DataFrame(data=d)

ax = df.plot.bar(x='DR_method', y='CV_AUC', rot=0)
plt.savefig('images/results2.png')

#%%
data = pd.read_csv('data/train.csv', parse_dates=['Date'])
X = data['Content_clean'].fillna('').values
y = data['ret_1-day'].fillna(1).values
y_pred = lda.predict_proba(X)

data['y_pred_0'] = y_pred[:,0]
data['y_pred_1'] = y_pred[:,1]
data['y_pred_2'] = y_pred[:,2]

data.to_csv('data/for_max/predictions.csv', index=False)
#%%
data2 = pd.read_csv('data/predictions.csv', parse_dates=['Date'])

#%%
plt.xlabel('DR_method')
plt.ylabel('CV_AUC')
plt.bar(df.DR_method, df.CV_AUC, width=0.5);

for i, v in enumerate(df.CV_AUC.values):
    ax.text(v + .0, i + .0, str(v), color='red', fontweight='bold')