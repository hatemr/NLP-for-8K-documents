import numpy as np
import spacy
print('importing nltkl')
import nltk
print('done')
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
from ey_nlp.contractions import CONTRACTION_MAP
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load('en_core_web_md', parse = True, tag=True, entity=True)
#nlp_vec = spacy.load('en_vecs', parse = True, tag=True, entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

#import ey_nlp.preprocessing

from . import preprocessing, model