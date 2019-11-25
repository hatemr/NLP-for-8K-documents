# -*- coding: utf-8 -*-

import pickle
import numpy as np
import numpy.random
import pandas as pd
import matplotlib.pyplot as plt

import scipy
#import imp
import time
import os
from IPython import embed

from ey_nlp.utils import dense_identity, drop_column

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin

from gensim.test.utils import common_dictionary, common_corpus
from gensim.sklearn_api import HdpTransformer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings('always')

#%%

if __name__ == "__main__":
    # read in data
    pickle_in = open("models/grid_search_ret_1-day_v2.pickle","rb")
    grid_search = pickle.load(pickle_in)
    pickle_in.close()
    
    # clean data for nice results
    d = grid_search.cv_results_
    results = pd.DataFrame(data=d)
    results = results.drop(columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params', 'mean_test_score', 'std_test_score'])
    
    # for groupby later
    dim_red_name = [r.__class__.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__dim_red']]
    #norm_name = ['' if r.__class__.__name__!='Normalizer' else r.__class__.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__norm']]
    rem_col_name = ['use_text_col' if not r else r.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__rem_col__func']]
    vect = [r.__class__.__name__ for r in  grid_search.cv_results_['param_preprocessor__text__vec']]
    clf_name = ['SGDClassifier' if r.__class__.__name__=='MaskedConstant' else r.__class__.__name__ for r in  grid_search.cv_results_['param_clf']]
    
    results['dim_red'] = dim_red_name
    results['rem_col'] = rem_col_name
    results['vect'] = vect
    results['clf'] = clf_name
    
    results1 = results.loc[:,['rank_test_score',
           'dim_red', 'rem_col', 'vect', 'clf','split0_test_score']]
    
    # to get max over within-model hyperparameters
    res1 = results1\
        .groupby(['rem_col', 'vect','dim_red','clf']).max().iloc[:,[-1]]\
        .round(3)\
        .reset_index()
    
    # for formatting md table for README.md
    df_lines = df2 = pd.DataFrame([['---',]*len(res1.columns)], columns=res1.columns) 
    
    df3 = pd.concat([df_lines, res1])
    
    # save as markdown
    df3.to_csv("data/res_ml_models.md", sep="|", index=False)
