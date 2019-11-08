#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 09:17:13 2019

@author: roberthatem
"""

import numpy as np
import pandas as pd
import scipy
#import imp
import pickle
import time
import os
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
#import ey_nlp
#imp.reload(ey_nlp)

#%%
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
    'Or is it the last?',
    'I am not sure.',
    'In statistics and machine learning, "model selection" is the problem of picking among different mathematical models which all purport to describe the same data set.',
    'This notebook will not (for now) give advice on it; as usual, it is more of a place to organize my thoughts and references.'
]
y = np.array([0,0,1,1,1,2,2,2])

#%%
pipeline1 = Pipeline([
    ('vec', CountVectorizer()),
    ('clf', SGDClassifier()),
])
parameters1 = {'vec__min_df': [0., 0.01],
               'clf__alpha': [0.00001, 0.000001]}
    
pipeline2 = Pipeline([
    ('vec', CountVectorizer()),
    ('dim_red', TruncatedSVD()), # new step
    ('clf', SGDClassifier()),
])
parameters2 = {'vec__min_df': [0., 0.01],
               'dim_red__n_components': [20, 50],
               'clf__alpha': [0.00001, 0.000001]}

# Combine
pipeline = Pipeline([
    ('vec', CountVectorizer()),
    ('clf', SGDClassifier()),
])
parameters = [
    {'vec__min_df': [0., 0.01],
     'clf': [SGDClassifier(),],
     'clf__alpha': (0.00001, 0.000001),
    },
    {'vec__min_df': [0., 0.01],
     'dim_red': [TruncatedSVD(),],  # new step?
     'dim_red__n_components': [20, 50], # new params?
     'clf': [SGDClassifier(),],
     'clf__alpha': (0.00001, 0.000001),
    }
]
grid_search = GridSearchCV(pipeline, parameters)
grid_search.fit(corpus, y)

#%%

pipeline = Pipeline([('vec', CountVectorizer()),
                     ('dim_red', FunctionTransformer(validate=True)),
                     ('norm', FunctionTransformer(validate=True)),
                     ('clf', SGDClassifier(loss='log', tol=1e-3))]
)
#%%
parameters = [
    {'vec__min_df': [0., 0.01],
     'dim_red': [TruncatedSVD()],
     'dim_red__n_components': [20, 50, 100],
     'norm': [Normalizer(copy=False)],
     'clf__alpha': [0.00001, 0.000001]}, 
    {
         'vec__min_df': [0., 0.01],
         'dim_red': [TruncatedSVD()],
         'dim_red__n_components': [20, 50, 100],
         'norm': [Normalizer(copy=False)],
         'clf': [RandomForestClassifier(random_state=0)],
         'clf__alpha': [0.00001, 0.000001],
         'clf__n_estimators': [100, 200],
         'clf__max_depth': [2,4],
         'clf__max_features': [2,'auto']
    }, 
{
         'vec__min_df': [0., 0.01],
         'dim_red': [LatentDirichletAllocation()],
         'dim_red__n_components': [5, 10],
         'clf__alpha': [0.00001, 0.000001]
    },
 {
         'vec__min_df': [0., 0.01],
         'dim_red': [LatentDirichletAllocation()],
         'dim_red__n_components': [5, 10],
         'clf': [RandomForestClassifier(random_state=0)],
         'clf__alpha': [0.00001, 0.000001],
         'clf__n_estimators': [100, 200],
         'clf__max_depth': [2,4],
         'clf__max_features': [2,'auto']
    }
]
#%%
grid_search = GridSearchCV(pipeline, parameters, cv=2)
grid_search.fit(corpus, y)

#%%

data = pd.read_csv('data/train.csv', parse_dates=['Date'])
data2 = pd.read_csv('data/test.csv', parse_dates=['Date'])
X = data['Content_clean'].fillna('').values
y = data['ret_1-day'].fillna(1).values

X1 = CountVectorizer().fit_transform(X).toarray()

pipeline = pipeline = Pipeline([('dim_red', FunctionTransformer(validate=True)),
                                ('norm', FunctionTransformer(validate=True))]
)

X2 = pipeline.fit_transform(X1)

#%%
import numpy as np
from sklearn.model_selection import PredefinedSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
test_fold = [-1, -1, -1, 0]
ps = PredefinedSplit(test_fold)
ps.get_n_splits()

print(ps)       

for train_index, test_index in ps.split():
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   
#%%
data = pd.DataFrame(data={'text_feat':['This is my first sentence. I hope you like it','Here is my second sentence. It is pretty similar.'], 
                          'numeric_feat':[1,0.8], 
                          'target':[3,4]})
X = data.loc[:,['text_feat', 'numeric_feat']]
y = data.loc[:,'target']
   

#def dense_identity(X):
#    return X.todense()
    
text_features = ['text_feat']
text_transformer = Pipeline(
        steps = [('vec', CountVectorizer())])#,
                 #('to_dense', FunctionTransformer(func=dense_identity, validate=True, accept_sparse=True))])

numeric_features = ['numeric_feat'] #['mkt_ret']
numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0.)),
               ('scaler', StandardScaler())])

# combine features preprocessing
preprocessor = ColumnTransformer(
        transformers=[('text', text_transformer, 'text_feat'),
                      ('num', numeric_transformer, numeric_features)])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

X1 = pipeline.fit_transform(X)
print('Expected (2, 13), got', X1.shape)

X2 = text_transformer.fit_transform(X['text_feat'])
print('Single pipeline works as expected:', X2.shape)

#%%
data = pd.DataFrame(data={'text_feat':['This is my first sentence.','This is my second.'], 
                          'numeric_feat':[1,2], 
                          'target':[3,4]})
X = data.loc[:,['text_feat', 'numeric_feat']]
y = data.loc[:,'target']
    
text_features = ['text_feat']
text_transformer = Pipeline(
        steps = [('vec', CountVectorizer())])

preprocessor = ColumnTransformer(
        transformers=[('text', text_transformer, 'text_feat')])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# single pipeline works as expected
X_expected = text_transformer.fit_transform(X['text_feat'])

X_test = pipeline.fit_transform(X)
print('Expected:')
print(X_expected.toarray())
print('Got:')
print(X_test)

#%%
