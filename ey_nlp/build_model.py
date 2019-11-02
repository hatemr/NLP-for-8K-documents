# -*- coding: utf-8 -*-

import pandas as pd
#import imp
import pickle
import time
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from gensim.test.utils import common_dictionary, common_corpus
from gensim.sklearn_api import HdpTransformer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#import ey_nlp
#imp.reload(ey_nlp)

#%%
def in_right_directory():
    dirpath = os.getcwd()
    foldername = os.path.basename(dirpath)
    return foldername == 'EY-NLP'

#%%
def save_model(model, filename = 'models/logreg.pickle'):
    '''
    Save model if it doesn't exists yet
    '''
    if in_right_directory():
        if os.path.exists(filename):
            print('File {} already exists. Not saved'.format(filename))
        else:
            file = open(filename, 'wb')
            pickle.dump(model, file)
            file.close()
            print('Saved file {}'.format(filename))
    else:
        print('To save the result in right directory, set \
              your current working directory to /EY-NLP')

#%%
def grid_search_func(X, y, pipeline, param_grid):
    '''
    Fit the GridSearchCV
    '''
    
    assert type(param_grid)==dict

    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(estimator = pipeline,
                               param_grid = param_grid,
                               scoring = 'f1_micro',
                               cv = tscv,
                               verbose=2)
    
    t0 = time.time()
    print("Performing grid search. This could take a while")
    grid_search.fit(X, y)
    print('Done fitting in {:.2f} minutes'.format((time.time()-t0)/60))
        
    return grid_search

#%%
def make_all_models():
    """Perform grid search on all combinations
    """
    
    data = pd.read_csv('data/train.csv', parse_dates=['Date'])
    data['Content_clean'] = data['Content_clean'].fillna('')
    data['2-day'] = data['2-day'].fillna(0.)
    X = data.loc[:, ['Content_clean','2-day']]
    y = data['ret_1-day'].fillna(0)
    
    '''
    # PCA, logreg
    step_pca_lr = Pipeline([('vec', CountVectorizer()),
                   ('svd', TruncatedSVD()),
                   ('norm', Normalizer(copy=False)),
                   ('clf', SGDClassifier(loss='log', tol=1e-3))])
    parameters_pca_lr = {'vec__min_df': (0., 0.1),
                         'svd__n_components': (20, 50, 100),
                         'clf__alpha': (0.00001, 0.000001)}
    
    # PCA, random forests
    step_pca_rf = Pipeline([('vec', CountVectorizer()),('svd', TruncatedSVD()),
                            ('norm', Normalizer(copy=False)),
                            ('clf', RandomForestClassifier(random_state=0))])
    parameters_pca_rf = {'vec__min_df': (0., 0.1),
                         'svd__n_components': (5, 20, 100),
                         'clf__n_estimators': (100, 150)}
    
    # LDA, logreg
    step_lda_lr = Pipeline([('vec', CountVectorizer()),
                            ('lda', LatentDirichletAllocation()),
                            ('clf', SGDClassifier(loss='log', tol=1e-3))])
    parameters_lda_lr = {'vec__min_df': (0., 0.1),
                         'lda__n_components': (5,10),
                         'clf__alpha': (0.00001, 0.000001)}
    
    step_lda_rf = Pipeline([('vec', CountVectorizer()),
                            ('lda', LatentDirichletAllocation()),
                            ('clf', RandomForestClassifier(random_state=0))])
    parameters_lda_rf = {#'vec__min_df': (0., 0.1),
                         'lda__n_components': [5,10],
                         'clf__n_estimators': [100, 200],
                         'clf__max_depth': [2,4],
                         'clf__max_features': [2,'auto']}
    
    # SESTM, logreg
    step_sestm_lr = Pipeline([('vec', CountVectorizer()),
                              ('clf', SGDClassifier(loss='log', tol=1e-3))])
    parameters_sestm_lr = {'vec__min_df': (0., 0.1),
                           'clf__alpha': (0.00001, 0.000001)}
    
    #HLDA
    text_transformer = Pipeline(steps = [('vec', CountVectorizer()),
                                         ('hlda', HdpTransformer(id2word=common_dictionary) )])

    '''
    
    # LDA, random forests
    text_features = ['Content_clean']
    text_transformer = Pipeline(
            steps = [('vec', CountVectorizer())])#,
                     #('lda', LatentDirichletAllocation())])
    
    numeric_features = ['2-day'] #['mkt_ret']
    numeric_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0.)),
                   ('scaler', StandardScaler())])
    
    # combine features preprocessing
    preprocessor = ColumnTransformer(
            transformers=[('text', text_transformer, text_features)])#,
                          #('num', numeric_transformer, numeric_features)])
    # add classifier
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=0))])

    X1 = text_transformer.fit_transform(data['Content_clean'])
    X2 = numeric_transformer.fit_transform(data['2-day'].values.reshape(-1,1))
    print(X1.shape, X2.shape)
    
    clf.fit(X, y)
    
    return 1
    
    
    #pipeline = Pipeline(steps=[('vec', CountVectorizer()),
    #                           ('clf', RandomForestClassifier(random_state=0))])
    #X = data.Content_clean.values
    #pipeline.fit(X, y)
    #print('hiiii')
    #vectorizer = CountVectorizer()
    #X = vectorizer.fit_transform(data)
    
    # param grid
    param_grid = {#'preprocessor__text__lda__n_components': [5,20],
                  #'classifier__n_estimators': [100,200],
                  #'classifier__max_depth': [2,4],
                  'classifier__max_features': [2,'auto']}

    
    models = [('lda_rf', clf, param_grid)]
              #('pca_lr', step_pca_lr, parameters_pca_lr), 
              #('lda_rf', step_lda_rf, parameters_lda_rf),
              #('sestm_lr', step_sestm_lr, parameters_sestm_lr)]

    for model in models:
        print('Starting {}'.format(model[0]))
        
        gs = grid_search_func(data, y, pipeline=model[1], param_grid=model[2])
        
        # save model. Use pickle + dictionaries
        model_name = model[0]
        model = {model_name: gs}
        
        filename = "models/" + model_name + ".pickle"
        save_model(model = model, filename = filename)
        
        print('Finished {}'.format(model_name))

#%%
if __name__ == "__main__":
    pipeline = make_all_models()
    
#%%
# Author: Pedro Morales <part.morales@gmail.com>
#
# License: BSD 3 clause

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)

# Read data from Titanic dataset.
titanic_url = ('https://raw.githubusercontent.com/amueller/'
               'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
data = pd.read_csv(titanic_url)

# We will train our classifier with the following features:
# Numeric Features:
# - age: float.
# - fare: float.
# Categorical Features:
# - embarked: categories encoded as strings {'C', 'S', 'Q'}.
# - sex: categories encoded as strings {'female', 'male'}.
# - pclass: ordinal integers {1, 2, 3}.

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_features = ['2-day']  #['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['ret_2-day'] #['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))]) #,
    #('onehot', OneHotEncoder(handle_unknown='ignore'))])

text_features = ['Content_clean']
text_transformer = Pipeline(
        steps=
        [('imputer', CountVectorizer(min_df=0.1))])
    
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', text_transformer, text_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

X = data.drop('survived', axis=1)
y = data['survived']

data = pd.read_csv('data/train.csv', parse_dates=['Date'])
#data['Content_clean'] = data['Content_clean'].fillna('')
#data['2-day'] = data['2-day'].fillna(0.)
X = data.loc[:, ['Content_clean', '2-day', 'ret_2-day']]
X['Content_clean'] = X['Content_clean'].fillna('')
y = data['ret_1-day'].fillna(0)
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))

#%%
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__C': [0.1, 1.0, 10, 100],
}

grid_search = GridSearchCV(clf, param_grid, cv=10, iid=False)
grid_search.fit(X_train, y_train)

print(("best logistic regression from grid search: %.3f"
       % grid_search.score(X_test, y_test)))
