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
    
    data = pd.read_csv('data/data.csv', parse_dates=['Date'])
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

    '''
    
    # LDA, random forests
    text_features = ['Content_clean']
    text_transformer = Pipeline(steps = [('vec', CountVectorizer()),
                                        ('lda', LatentDirichletAllocation())])
    numeric_features = [''] #['mkt_ret']
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0.)),
                                          ('scaler', StandardScaler())])
    
    # combine features preprocessing
    preprocessor = ColumnTransformer(transformers=[('text', text_transformer, text_features),
                                                   ('num', numeric_transformer, numeric_features)])
    # add classifier
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=0))])
    # param grid
    param_grid = {'preprocessor__text__lda__n_components': (0., 0.1),
                  'classifier__n_estimators': [100, 200],
                  'classifier__max_depth': [2,4],
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
    