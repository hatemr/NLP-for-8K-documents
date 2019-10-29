# -*- coding: utf-8 -*-

import pandas as pd
import imp
import pickle
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import ey_nlp
#imp.reload(ey_nlp)

#%%
df = pd.read_csv('data/8ks_with_returns.csv', parse_dates=[1])

#%%
# get text
corpus = df.Content.values.tolist()

#%%
# clean text
t0 = time.time()
corpus_cleaned = [ey_nlp.preprocessing.preprocess_text(doc) for doc in corpus]
print(time.time() - t0)

#%%
# tockenize
t0 = time.time()
corpus_tokenized = [ey_nlp.preprocessing.custom_tokenizer(doc) for doc in corpus_cleaned]
print(time.time()-t0)

#%%
df['Content_clean'] = corpus_tokenized

#%%
cols = df.columns.tolist()
cols2 = cols[:3] + cols[-1:] + cols[3:-1]

df = df[cols2]

#%%
df.to_csv('data/8ks_with_returns_cleaned.csv')

#%%
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
    
vocab = vectorizer.get_feature_names()
doc_term_mat = X.toarray()

#%%
# make document-term matrix with a list of its words
vocab, doc_term_mat = ey_nlp.preprocessing.count_words(corpus)

#%%



#%%
data = {'doc_term_mat': doc_term_mat, 'vocab': vocab}

pickle_out = open("data.pickle","wb")
pickle.dump(data, pickle_out)
pickle_out.close()

#%%