# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#import imp
import pickle
import time
import os
from IPython import embed
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin

from spacy.lang.en import English # updated

from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch

from gensim.test.utils import common_dictionary, common_corpus
from gensim.sklearn_api import HdpTransformer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings('always')
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
        if os.path.exists(filename): #False
            print('File {} already exists. Not saved'.format(filename))
        else:
            file = open(filename, 'wb')
            pickle.dump(model, file)
            file.close()
            print('Saved file {}'.format(filename))
    else:
        print('To save the result in right directory, set \
              your current working directory to /EY-NLP')

def dense_identity(X):
    try:
        return X.todense()
    except:
        return X

def drop_BOW_feat(X):
    ones = np.ones(X.shape[0])
    return ones[:, np.newaxis]

def drop_BERT_feat(X):
    """ Assumes sentiment is first column """
    return X[:,[0]] #.loc[:,['sentiment']]

def remove_extra_newlines(text):
    return re.sub(r'[\r|\n|\r\n]+', ' ', text)

#%%
def make_all_models(target='ret_1-day'):
    """Perform grid search on all combinations
    """
    
    data = pd.read_csv('data/train_bert.csv', parse_dates=['Date'])
    data['Content_clean'] = data['Content_clean'].fillna('')
    data['sentiment_x'] = data['sentiment_x'].fillna(0)
    data[target] = data[target].fillna(1)
    
    cols = ['Date','Ticker','Content_clean','sentiment_x'] + ['bert_' + str(i) for i in range(767)]
    X = data.loc[:,cols]
    #X = data['Content_clean'].fillna('').values
    #y = data['ret_1-day'].fillna(1).values
    y = data[target]
    
    
#    text_features = ['Content_clean']
    text_transformer = Pipeline(
            steps = [#('text_imp', SimpleImputer(strategy='constant', fill_value='')),
                     ('vec', CountVectorizer()),
                     ('dim_red', FunctionTransformer(func=dense_identity, validate=True, accept_sparse=True)),
                     ('norm', FunctionTransformer(func=dense_identity, validate=True, accept_sparse=True)),
                     ('rem_col', FunctionTransformer(func=None, validate=True, accept_sparse=True))
                     ])


    numeric_features = ['sentiment_x'] + ['bert_' + str(i) for i in range(767)]
    numeric_transformer = Pipeline(
            steps=[('rem_col', FunctionTransformer(func=None, validate=True, accept_sparse=True)),
                   ('num_imp', SimpleImputer(strategy='constant', fill_value=0.)),
                   ('scaler', StandardScaler())])

    # combine features preprocessing
    preprocessor = ColumnTransformer(
            transformers=[('text', text_transformer, 'Content_clean'),
                          ('num', numeric_transformer, numeric_features)])
    # add classifier
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('clf', SGDClassifier(loss='log', tol=1e-3))])

    
    parameters = [
        {
            'preprocessor__text__rem_col__func': [drop_BOW_feat], #None, drop_BOW_feat
            'preprocessor__text__vec': [CountVectorizer()],
            'preprocessor__text__vec__min_df': [0., 0.01, 0.02],
            'preprocessor__text__dim_red': [TruncatedSVD()],
            'preprocessor__text__dim_red__n_components': [100, 200], #100
            'preprocessor__text__norm': [Normalizer(copy=False)],
            'preprocessor__num__rem_col__func': [None], #None, drop_BERT_feat
            'clf__alpha': [0.00005, 0.000001, 0.000005]
        }
    ]

    # predefined split lets us use one validation set (instead of multiple in cv)
    test_fold = np.zeros(y.shape)
    test_fold[0:9947] = -1
    ps = PredefinedSplit(test_fold=test_fold)
    
    grid_search = GridSearchCV(estimator = clf,
                               param_grid = parameters,
                               scoring = 'f1_weighted',
                               cv = ps,
                               verbose=2,
                               n_jobs=1)
    
    t0 = time.time()
    print("Performing grid search. This could take a while")
    grid_search.fit(X, y)
    print('Done fitting in {:.2f} minutes'.format((time.time()-t0)/60))
    
    # save model. Use pickle + dictionaries
    model_name = 'grid_search_' + horizon + '_bert_v7'
    
    filename = "models/" + model_name + ".pickle"
    save_model(model = grid_search, filename = filename)
    
    return grid_search

#%%
if __name__ == "__main__":
            
    for h in ['1-day']:#, '2-day', '3-day', '5-day', '10-day', '20-day', '30-day']:
        horizon = 'ret_' + h
        print('starting', horizon)
        grid_search = make_all_models(target='alpha_1-day')
        print('done', horizon)
        embed()
        print('done')
#%%






#%%
def make_results_dataframe(grid_search):
    '''Find best model class in the grid_search
    '''
    
    d = grid_search.cv_results_
    results = pd.DataFrame(data=d)
    results = results.drop(columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params', 'mean_test_score', 'std_test_score'])

    # for groupby later
    dim_red_name = [r.__class__.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__dim_red']]
    norm_name = ['' if r.__class__.__name__!='Normalizer' else r.__class__.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__norm']]
    rem_col_name = ['' if r.__class__.__name__!='drop_column' else r.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__rem_col__func']]
    clf_name = ['SGDClassifier' if r.__class__.__name__=='MaskedConstant' else r.__class__.__name__ for r in  grid_search.cv_results_['param_clf']]
    
    results['dim_red_name'] = dim_red_name
    results['norm_name'] = norm_name
    results['rem_col_name'] = rem_col_name
    results['clf_name'] = clf_name
    
    # find max within model class
    results.groupby(['dim_red_name', 'norm_name', 'clf_name']).max().loc[:,'mean_test_score']
    
    results.to_csv('data/results2.csv', index=False)
