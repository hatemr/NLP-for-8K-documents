# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
#%%
df = pd.read_csv('8ks_with_returns_cleaned_v2.csv', parse_dates=[2])
#%%
X = df['Content_clean'].values
y = df.target.values
# y = 0,1,2
#%%
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

#%%
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
X = svd.fit_transform(X)

#%%
normalizer = Normalizer(copy=False)
X = normalizer.fit_transform(X)
print(type(X))
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0, max_features=2)
# print(clf.fit(X, y))

#%%
# Data is ready for ML
# Bonus: can you combine the above preprocessing steps using sklearn's Pipeline?
# print(X.shape, y.shape)
tscv = TimeSeriesSplit(n_splits=3)
parameter_grid = {'n_estimators':[100, 150, 200], 'max_depth':[2,3,4], 'random_state':[0], 'max_features':[1,2,3]}
clf_2 = GridSearchCV(clf, parameter_grid, cv = tscv)
clf_2.fit(X,y)
print(clf_2.cv_results_)
