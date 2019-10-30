lo# -*- coding: utf-8 -*-

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
def make_model(corpus, y, step, parameters):
    '''
    Train the logistic regression using gridsearch
    '''
    
    assert type(step)==list
    assert type(parameters)==dict
    
    step1 = [('vec', CountVectorizer())]
    step2 = [('clf', SGDClassifier(loss='log', tol=1e-3))]
    steps = step1 + step + step2
    
    pipeline = Pipeline(steps = steps)
#    pipeline = Pipeline(steps = [
#        ('vec', CountVectorizer()), 
#        ('svd', TruncatedSVD()), 
#        ('norm', Normalizer(copy=False)),
#        ('clf', SGDClassifier(loss='log', tol=1e-3))])
    
    params1 = {'vec__min_df': (0., 0.1)} #,
               #'clf__alpha': (0.00001, 0.000001)}
    
    params = {**params1, **parameters}
        
    tscv = TimeSeriesSplit(n_splits=2)
    
    grid_search = GridSearchCV(estimator = pipeline,
                               param_grid = params,
                               scoring = 'roc_auc',
                               cv = tscv,
                               verbose=4)
    
    t0 = time.time()
    print("Performing grid search. ~5 min")
    grid_search.fit(corpus, y)
    print('Done fitting in {:.2f} minutes'.format((time.time()-t0)/60))
    
    # save model. Use pickle + dictionaries
    model_name = step[0][0]
    model = {model_name: grid_search}
    
    filename = "models/" + model_name + ".pickle"
    save_model(model = model, filename = filename)
        
    return grid_search

#%%
def make_all_models():
    df = pd.read_csv('data/8ks_with_returns_cleaned.csv', parse_dates=[1])
    
    # extract list of documents (strings)
    corpus = df.Content.values.tolist()
    y = (df['1-day'] > 0).astype(int).values
    
    # PCA
    step_pca = [('svd', TruncatedSVD()),
                ('norm', Normalizer(copy=False))]
    parameters_pca = {'svd__n_components': (5, 20, 100)}
    
    # LDA
    step_lda = [('lda', LatentDirichletAllocation())]
    parameters_lda = {'lda__n_components': (5,10)}
    
    # SESTM
    step_sestm = []
    parameters_sestm = {}
    
    models = [#(step_pca, parameters_pca), 
              (step_lda, parameters_lda)] #,
              #(step_sestm, parameters_sestm)]

    for model in models:
        print('Starting {}'.format(model[0][0][0]))
        gs = make_model(corpus, y, step=model[0], parameters=model[1])
        print('Finished {}'.format(model[0][0][0]))
        

#%%
if __name__ == "__main__":
    make_all_models()
