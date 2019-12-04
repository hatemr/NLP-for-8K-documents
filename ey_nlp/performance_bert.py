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
pickle_in = open("models/grid_search_ret_1-day_bert_v1.pickle","rb")
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
text_rem_col_name = ['use_BOW_feat' if not r else r.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__rem_col__func']]
num_rem_col_name = ['use_BERT_feat' if not r else r.__name__ for r in  grid_search.cv_results_['param_preprocessor__num__rem_col__func']]


results['text_rem_col'] = text_rem_col_name
results['num_rem_col'] = num_rem_col_name

results1 = results.loc[:,['rank_test_score',
    'text_rem_col', 'num_rem_col', 'split0_test_score']]

res1 = results1\
    .groupby(['text_rem_col', 'num_rem_col']).max().iloc[:,[-1]]\
    .round(3)\
    .reset_index()

df_lines = pd.DataFrame([['---',]*len(res1.columns)], columns=res1.columns) 

df3 = pd.concat([df_lines, res1])

# save as markdown
#df3.to_csv("data/res_bert_models.md", sep="|", index=False)
df3
#%%
# test for Max
pickle_in = open("models/grid_search_ret_1-day_bert_v1.pickle","rb")
grid_search = pickle.load(pickle_in)
pickle_in.close()

target = 'ret_1-day'
data = pd.read_csv('data/test_bert.csv', parse_dates=['Date'])
#data['Content_clean'] = data['Content_clean'].fillna('')
data['sentiment_x'] = data['sentiment_x'].fillna(0)
#data[target] = data[target].fillna(1)
    
cols = ['Date','Ticker','Content_clean','sentiment_x'] + ['bert_' + str(i) for i in range(767)]
X = data.loc[:,cols]

y_pred = grid_search.predict_proba(X)
print(y_pred.shape)

#%%
data = pd.read_csv('data/test_bert.csv', parse_dates=['Date'])
with open('data/test_bert.pickle', 'wb') as handle:
    pickle.dump(data, handle)
