# -*- coding: utf-8 -*-

import pickle
import numpy as np
import numpy.random
import pandas as pd
import matplotlib.pyplot as plt

import scipy
#import imp
import time
import os
from IPython import embed

from ey_nlp.utils import dense_identity, drop_column

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin

from gensim.test.utils import common_dictionary, common_corpus
from gensim.sklearn_api import HdpTransformer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings('always')

#%%
# read in data
pickle_in = open("models/grid_search_ret_1-day_v2.pickle","rb")
grid_search = pickle.load(pickle_in)
pickle_in.close()

# clean data for nice results
d = grid_search.cv_results_
results = pd.DataFrame(data=d)
results = results.drop(columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params', 'mean_test_score', 'std_test_score'])

#d = grid_search.cv_results_
#results = pd.DataFrame(data=d)
#results = results.drop(columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params', 'mean_test_score', 'std_test_score'])

# for groupby later
dim_red_name = [r.__class__.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__dim_red']]
#norm_name = ['' if r.__class__.__name__!='Normalizer' else r.__class__.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__norm']]
rem_col_name = ['use_text_col' if not r else r.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__rem_col__func']]
vect = [r.__class__.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__vec']]
clf_name = ['SGDClassifier' if r.__class__.__name__=='MaskedConstant' else r.__class__.__name__ for r in  grid_search.cv_results_['param_clf']]

results['dim_red'] = dim_red_name
results['rem_col'] = rem_col_name
results['vect'] = vect
results['clf'] = clf_name

results1 = results.loc[:,['rank_test_score',
       'dim_red', 'rem_col', 'vect', 'clf','split0_test_score']]

res1 = results1\
    .groupby(['rem_col', 'vect','dim_red','clf']).max().iloc[:,[-1]]\
    .round(3)\
    .reset_index()

df_lines = df2 = pd.DataFrame([['---',]*len(res1.columns)], columns=res1.columns) 

df3 = pd.concat([df_lines, res1])

# save as markdown
df3.to_csv("res_ml_models.md", sep="|", index=False)

#%%
results[['param_preprocessor__text__dim_red__n_components', 'split0_test_score']] == 0.469760 


#%%
dim_red = ['TruncatedSVD', 'LatentDirichletAllocation'][0]
rem_col = ['drop_column', 'use_text_col'][0]
vect = ['CountVectorizer', 'TfidfVectorizer'][0]
clf = ['SGDClassifier', 'RandomForestClassifier'][0]

group_by = ['dim_red', 'rem_col', 'vect','clf'][1]

# adding text improves performance
results1.loc[(results1.dim_red==dim_red) & (results1.vect==vect) & (results1.clf==clf)].groupby(group_by).max().iloc[:,[-1]]

#%%
dim_red = ['TruncatedSVD', 'LatentDirichletAllocation'][1]
rem_col = ['drop_column', 'use_text_col'][0]
vect = ['CountVectorizer', 'TfidfVectorizer'][0]
clf = ['SGDClassifier', 'RandomForestClassifier'][0]

group_by = ['dim_red', 'rem_col', 'vect','clf'][1]

# but not as much with LDA
results1.loc[(results1.dim_red==dim_red) & (results1.vect==vect) & (results1.clf==clf)].groupby(group_by).max().iloc[:,[-1]]

#%%
dim_red = ['TruncatedSVD', 'LatentDirichletAllocation'][0]
#rem_col = ['drop_column', 'use_text_col'][0]
vect = ['CountVectorizer', 'TfidfVectorizer'][0]
clf = ['SGDClassifier', 'RandomForestClassifier'][1]

group_by = ['dim_red', 'rem_col', 'vect','clf'][1]

# random forests now: adding text still helps
results1.loc[(results1.dim_red==dim_red) & (results1.vect==vect) & (results1.clf==clf)].groupby(group_by).max().iloc[:,[-1]]

#%%
# compare classifiers
dim_red = ['TruncatedSVD', 'LatentDirichletAllocation'][0]
rem_col = ['drop_column', 'use_text_col'][0]
vect = ['CountVectorizer', 'TfidfVectorizer'][0]
clf = ['SGDClassifier', 'RandomForestClassifier'][1]

group_by = ['dim_red', 'rem_col', 'vect','clf'][1]

# random forests now: adding text still helps
results1.loc[(results1.rem_col==rem_col) \
             & (results1.vect==vect) \
             ]\
             .groupby(['dim_red', 'clf']).max().iloc[:,[-1]]

#%%
# compare classifiers AND dimensionality reduction
dim_red = ['TruncatedSVD', 'LatentDirichletAllocation'][0]
rem_col = ['drop_column', 'use_text_col'][0]
vect = ['CountVectorizer', 'TfidfVectorizer'][1]
clf = ['SGDClassifier', 'RandomForestClassifier'][1]

group_by = ['dim_red', 'rem_col', 'vect','clf'][1]

# 2-by-2
results1.loc[(results1.rem_col==rem_col) \
             & (results1.vect==vect)]\
             .groupby(['dim_red', 'clf']).max().iloc[:,[-1]]
             


             
#%%
data = pd.read_csv('data/test.csv', parse_dates=['Date'])

X_test = data
embed()
#%%


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

#%%
data1 = data.iloc[:, [0,1,2,19,20,21]]

data1.to_csv('data/for_max/data.csv', index=False)
#%%
data2 = pd.read_csv('data/for_max/data.csv', parse_dates=['Date'])

#%%
plt.xlabel('DR_method')
plt.ylabel('CV_AUC')
plt.bar(df.DR_method, df.CV_AUC, width=0.5);

for i, v in enumerate(df.CV_AUC.values):
    ax.text(v + .0, i + .0, str(v), color='red', fontweight='bold')
    
#%%
# make test predictions for Max (11/6/19)
pickle_in = open("models/grid_search.pickle","rb")
file = pickle.load(pickle_in)
pickle_in.close()

grid_search = file['grid_search']

#%%
test = pd.read_csv('data/test.csv', parse_dates=['Date'])

#%%
X_test = test['Content_clean'].fillna('').values
y_pred = grid_search.predict_proba(X_test)

#%%
y_test = test.loc[:,['Date','Ticker']]
y_test['y_pred_0'] = y_pred[:,0]
y_test['y_pred_1'] = y_pred[:,1]
y_test['y_pred_2'] = y_pred[:,2]

#%%
y_test.to_csv('data/y_pred_oos.csv', index=False)

#%% 11/23/19
data = pd.read_csv('data/train.csv', parse_dates=['Date'])
embed()