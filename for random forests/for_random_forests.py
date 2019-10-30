# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

#%%
#df = pd.read_csv('8ks_with_returns_cleaned_v2.csv', parse_dates=[2])

df = pd.read_csv('8ks_with_returns_cleaned.csv', parse_dates=[2])

#%%
# discretize target variable
d = df['1-day'].fillna(0.0).values
y_up = 2*(d >= 0.01).astype(int)
y_mid = 1*((d<0.01)&(d>-0.01)).astype(int)
y_down = 0*(d <= -0.01).astype(int)
y = y_up + y_mid + y_down

#%%
X = df['Content_clean'].values

#%%
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

#%%
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
X = svd.fit_transform(X)

#%%
normalizer = Normalizer(copy=False)
X = normalizer.fit_transform(X)

#%%
# Data is ready for ML
# Bonus: can you combine the above preprocessing steps using sklearn's Pipeline?
print(X.shape, y.shape)