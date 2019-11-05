# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#import imp
import pickle
import time
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
#from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

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
                               scoring = 'f1_weighted',
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
    X = data['Content_clean'].fillna('').values
    y = data['ret_1-day'].fillna(1).values
    
    
    '''
    # LDA, random forests
    text_features = ['Content_clean']
    text_transformer = Pipeline(
            steps = [('vec', CountVectorizer()),
                     ('lda', LatentDirichletAllocation()),
                     ('clf', RandomForestClassifier(random_state=0))])
    
#    numeric_features = ['2-day'] #['mkt_ret']
#    numeric_transformer = Pipeline(
#            steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0.)),
#                   ('scaler', StandardScaler())])
    
    # combine features preprocessing
    preprocessor = ColumnTransformer(
            transformers=[('text', text_transformer, text_features)])#,
                          #('num', numeric_transformer, numeric_features)])
    # add classifier
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=0))])

    # param grid
#    param_grid = {#'preprocessor__text__lda__n_components': [5,20],
#                  #'classifier__n_estimators': [100,200],
#                  #'classifier__max_depth': [2,4],
#                  'classifier__max_features': [2,'auto']}

    param_grid_lda_rf = {'vec__max_features': [5000, 10000],
                         'lda__n_components': [5, 10],
                         'clf__n_estimators': [100, 200],
                         'clf__max_depth': [2,4],
                         'clf__max_features': ['auto']} #2

    '''
    
    pipeline = Pipeline([('vec', CountVectorizer()),
                     ('dim_red', FunctionTransformer(validate=True)),
                     ('norm', FunctionTransformer(validate=True)),
                     ('clf', SGDClassifier(loss='log', tol=1e-3))]
    )
    parameters = [
        {
            'vec__min_df': [0., 0.01],
            'dim_red': [TruncatedSVD()],
            'dim_red__n_components': [20, 50, 100],
            'norm': [Normalizer(copy=False)],
            'clf__alpha': [0.0001, 0.00001, 0.000001]
        }, {
             'vec__min_df': [0., 0.01],
             'dim_red': [TruncatedSVD()],
             'dim_red__n_components': [20, 50, 100],
             'norm': [Normalizer(copy=False)],
             'clf': [RandomForestClassifier(random_state=0)],
             'clf__n_estimators': [100, 200],
             'clf__max_depth': [2,4],
             'clf__max_features': [2,'auto']
        }, {
             'vec__min_df': [0., 0.01],
             'dim_red': [LatentDirichletAllocation()],
             'dim_red__n_components': [5, 10],
             'clf__alpha': [0.0001, 0.00001, 0.000001]
        }, {
             'vec__min_df': [0., 0.01],
             'dim_red': [LatentDirichletAllocation()],
             'dim_red__n_components': [5, 10],
             'clf': [RandomForestClassifier(random_state=0)],
             'clf__n_estimators': [100, 200],
             'clf__max_depth': [2,4],
             'clf__max_features': [2,'auto']
        }
    ]

    #tscv = TimeSeriesSplit(n_splits=2)
    
    test_fold = np.zeros(y.shape)
    test_fold[0:9947] = -1
    # predefined split lets us use one validation set (instead of multiple in cv)
    ps = PredefinedSplit(test_fold=test_fold)
    
    grid_search = GridSearchCV(estimator = pipeline,
                               param_grid = parameters,
                               scoring = 'f1_weighted',
                               cv = ps,
                               verbose=2)
    
    t0 = time.time()
    print("Performing grid search. This could take a while")
    grid_search.fit(X, y)
    print('Done fitting in {:.2f} minutes'.format((time.time()-t0)/60))
    
    # save model. Use pickle + dictionaries
    model_name = 'grid_search'
    model = {model_name: grid_search}
    
    filename = "models/" + model_name + ".pickle"
    save_model(model = model, filename = filename)
    
    return grid_search
    
    '''
    #('lda_rf_2', text_transformer, param_grid_lda_rf)

    models = [#('pca_lr', pipeline_pca_lr, param_grid_pca_lr), 
              #('pca_rf', pipeline_pca_rf, param_grid_pca_rf)#,
              #('lda_lr', pipeline_lda_lr, param_grid_lda_lr)#,
              ('lda_rf', pipeline_lda_rf, param_grid_lda_rf)]

    model_names = []
    best_score = []

    for model in models:
        print('Starting {}'.format(model[0]))
        
        gs = grid_search_func(X, y, pipeline=model[1], param_grid=model[2])
        
        # save model. Use pickle + dictionaries
        model_name = model[0]
        model = {model_name: gs}
        
        filename = "models/" + model_name + ".pickle"
        save_model(model = model, filename = filename)
        
        print('Finished {}'.format(model_name))
        
        model_names.append(model_name)
        best_score.append(gs.best_score_)
        print('Best score: {}'.format(gs.best_score_))
        
    d = {'model': model_name, 'best_f1_weighted':best_score}
    results = pd.DataFrame(data=d)
    return results
    '''

#%%
if __name__ == "__main__":
    grid_search = pipeline = make_all_models()

#%%
d = grid_search.cv_results_
results = pd.DataFrame(data=d)
#%%
# 
dim_red_name = [r.__class__.__name__ for r in  grid_search.cv_results_['param_dim_red']]
norm_name = ['' if r.__class__.__name__!='Normalizer' else r.__class__.__name__ for r in  grid_search.cv_results_['param_norm']]
clf_name = ['SGDClassifier' if r.__class__.__name__=='MaskedConstant' else r.__class__.__name__ for r in  grid_search.cv_results_['param_clf']]

results['dim_red_name'] = dim_red_name
results['norm_name'] = norm_name
results['clf_name'] = clf_name

#%%
results.groupby(['dim_red_name', 'norm_name', 'clf_name']).max().loc[:,'mean_test_score']

#%%

def get_lda_topics():
    '''Creates files in /topics folder for Chelsea (11/5/19)
    '''
    
    data = pd.read_csv('data/train.csv', parse_dates=['Date'])
    X = data['Content_clean'].fillna('').values
    
    vectorizer = CountVectorizer(max_features=10000)
    vectorizer.fit(X)
    X1 = vectorizer.transform(X)
    
    lda = LatentDirichletAllocation(n_components=10)
    lda.fit(X1)
    X2 = lda.transform(X1)
    
    topic_vectors = lda.components_
    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    
    df = pd.DataFrame(topic_vectors.T)
    df['vocab'] = vocab
    
    df.to_csv('topics/topic_vectors.csv', index=False)

    data2 = data.loc[:,['Date', 'Ticker', 'Content', 'Content_clean']]
    
    data3 = data2.join(pd.DataFrame(
        {
            'topic_1': X2[:,0],
            'topic_2': X2[:,1],
            'topic_3': X2[:,2],
            'topic_4': X2[:,3],
            'topic_5': X2[:,4],
            'topic_6': X2[:,5],
            'topic_7': X2[:,6],
            'topic_8': X2[:,7],
            'topic_9': X2[:,8],
            'topic_10': X2[:,9],
        }, index=data2.index
    ))
    data3.to_csv('topics/topics.csv', index=False)
    
#%%