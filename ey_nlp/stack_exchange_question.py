#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 09:17:13 2019

@author: roberthatem
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer


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
   