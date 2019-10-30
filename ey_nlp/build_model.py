# -*- coding: utf-8 -*-

import pandas as pd
#import imp
import pickle
import time
import os
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.decomposition import LatentDirichletAllocation
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
            file = open('models/logreg.pickle', 'wb')
            pickle.dump(model, file)
            file.close()
            print('Saved file {}'.format(filename))
    else:
        print('To save the result in right location, set \
              your current working directory to /EY-NLP')

#%%
def make_pca(corpus, y):
    '''
    Train the logistic regression using gridsearch
    '''
    
    pipeline = Pipeline(steps = [
        ('vec', CountVectorizer()), 
        ('svd', TruncatedSVD()), 
        ('norm', Normalizer(copy=False)),
        ('clf', SGDClassifier(loss='log', tol=1e-3))])
    
    parameters = {
            'vec__min_df': (0.01, 0.1),
            'svd__n_components': (5, 20, 100),
            'clf__alpha': (0.00001, 0.000001)
            }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(estimator = pipeline,
                               param_grid = parameters,
                               scoring = 'roc_auc',
                               cv = tscv)
    
    t0 = time.time()
    print("Performing grid search. ~5 min")
    grid_search.fit(corpus, y)
    print('Done fitting in {:.2f} minutes'.format((time.time()-t0)/60))
    
    # save model. Use pickle + dictionaries
    model = {'logreg': grid_search}
    
    save_model(model = model, filename = 'models/logreg.pickle')
        
    return grid_search

#%%
if __name__ == "__main__":
    # import data
    df = pd.read_csv('data/8ks_with_returns_cleaned.csv', parse_dates=[1])
    
    # extract list of documents (strings)
    corpus = df.Content.values.tolist()
    y = (df['1-day'] > 0).astype(int).values
    
    logreg = make_pca(corpus, y)
