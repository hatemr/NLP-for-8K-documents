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
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

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
def grid_search_func(corpus, y, pipeline, param_grid):
    '''
    Fit the GridSearchCV
    '''
    
    assert type(pipeline)==list
    assert type(param_grid)==dict

    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(estimator = pipeline,
                               param_grid = param_grid,
                               scoring = 'roc_auc',
                               cv = tscv,
                               verbose=2)
    
    t0 = time.time()
    print("Performing grid search. ~5 min")
    grid_search.fit(corpus, y)
    print('Done fitting in {:.2f} minutes'.format((time.time()-t0)/60))
        
    return grid_search

#%%
def make_all_models():
    df = pd.read_csv('data/data.csv', parse_dates=['Date'])
    
    # extract list of documents (strings)
    corpus = df.Content.values.tolist()
    d = df['1-day'].fillna(0).values
    
    y_up = 2*(d >= 0.01).astype(int)
    y_mid = 1*((d<0.01)&(d>-0.01)).astype(int)
    y_down = 0*(d <= -0.01).astype(int)
    
    y = y_up + y_mid + y_down
    
    # PCA, logreg
    step_pca_lr = Pipeline([('vec', CountVectorizer()),
                   ('svd', TruncatedSVD()),
                   ('norm', Normalizer(copy=False)),
                   ('clf', SGDClassifier(loss='log', tol=1e-3))])
    parameters_pca_lr = {'vec__min_df': (0., 0.1),
                         'svd__n_components': (5, 20, 100),
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

    # LDA, random forests
    step_lda_rf = Pipeline([('vec', CountVectorizer()),
                            ('lda', LatentDirichletAllocation()),
                            ('clf', RandomForestClassifier(random_state=0))])
    parameters_lda_rf = {'vec__min_df': (0., 0.1),
                         'lda__n_components': (5,10),
                         'clf__n_estimators': (100, 150)}
    
    # SESTM, logreg
    step_sestm_lr = Pipeline([('vec', CountVectorizer()),
                              ('clf', SGDClassifier(loss='log', tol=1e-3))])
    parameters_sestm_lr = {'vec__min_df': (0., 0.1),
                           'clf__alpha': (0.00001, 0.000001)}
    
    models = [('pca_lr', step_pca_lr, parameters_pca_lr), 
              ('lda_lr', step_lda_lr, parameters_lda_lr),
              ('sestm_lr', step_sestm_lr, parameters_sestm_lr)]

    for model in models:
        print('Starting {}'.format(model[0]))
        gs = grid_search_func(corpus, y, pipeline=model[1], param_grid=model[2])
        
        # save model. Use pickle + dictionaries
        model_name = model[0]
        model = {model_name: gs}
        
        filename = "models/" + model_name + ".pickle"
        save_model(model = model, filename = filename)
        
        print('Finished {}'.format(model[0]))
        
#%%
if __name__ == "__main__":
    make_all_models()
    