# -*- coding: utf-8 -*-

# Taken largely from this website
# https://www.kdnuggets.com/2018/08/practitioners-guide-processing-understanding-text-2.html

import pandas as pd
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
import time

nlp = spacy.load('en_core_web_md', parse = True, tag=True, entity=True)
#nlp_vec = spacy.load('en_vecs', parse = True, tag=True, entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

#%%
# from the paper
def remove_proper_nouns(text = 'I am named John Dow'):
    # written by Robert Hatem
    text_tagged = nltk.tag.pos_tag(text.split())
    text_edited = [word for word, tag in text_tagged if tag != 'NNP' and tag != 'NNPS']
    text_new = ' '.join(text_edited)
    return text_new

def lower_case(text):
    return text.lower()

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


# extra preprocessing
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_extra_newlines(text):
    return re.sub(r'[\r|\n|\r\n]+', ' ', text)
    
def remove_extra_whitespace(text):
    return re.sub(' +', ' ', text)


# for customer tockenizer
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

#%%
def preprocess_text(doc,
                    proper_noun_removal=True,
                    lower_the_case=True,
                    contraction_expansion=True,
                    special_char_removal=True,
                    remove_digits=True,
                    stopword_removal=True,
                    html_stripping=True,
                    accented_char_removal=True):
    """
    doc: string
    """
    
    # remove proper nouns
    if proper_noun_removal:
        doc = remove_proper_nouns(doc)
    # lowercase the text    
    if lower_the_case:
        doc = lower_case(doc)
    # expand contractions    
    if contraction_expansion:
        doc = expand_contractions(doc)
    if special_char_removal:
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        doc = remove_special_characters(doc, remove_digits=remove_digits)
    # remove stopwords
    if stopword_removal:
        doc = remove_stopwords(doc, is_lower_case=lower_the_case)
    # strip HTML
    if html_stripping:
        doc = strip_html_tags(doc)
    # remove accented characters
    if accented_char_removal:
        doc = remove_accented_chars(doc)
    doc = remove_extra_newlines(doc)
    doc = remove_extra_whitespace(doc)
        
    return doc

#%%
def custom_tokenizer(doc,
                     stemmer=True,
                     text_lemmatization=True):
    # stemmer
    if stemmer:
        doc = simple_stemmer(doc)
    # lemmatize text
    if text_lemmatization:
        doc = lemmatize_text(doc)
    return doc.split()

#%%
def clean_text(doc):
    """
    Clean and tockenize data
    """
    doc_clean = preprocess_text(doc)
    doc_clean = custom_tokenizer(doc_clean)
    return doc_clean

#%%
def _count_words(corpus):
    '''
    Makes document-term matrix
    
    corpus: list of strings
    
    reference: https://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
    '''
    
    
    
    if type(corpus) != list:
        raise TypeError('corpus should be list or nltk corpus')
        
    vectorizer = CountVectorizer(preprocessor=preprocess_text,
                                 tokenizer=custom_tokenizer)
    X = vectorizer.fit_transform(corpus)
    
    vocab = vectorizer.get_feature_names()
    doc_term_mat = X.toarray()

    return vocab, doc_term_mat
    
#%%
if __name__ == "__main__":
    df = pd.read_csv('data/8ks_with_returns.csv', parse_dates=[1])
    
    # get text
    corpus = df.Content.values.tolist()
    
    # clean text
    print('Cleaning text... \nThis could take 30 minutes')
    
    t0 = time.time()
    corpus_cleaned = [preprocess_text(doc) for doc in corpus]
    
    # tockenize
    t0 = time.time()
    corpus_tokenized = [custom_tokenizer(doc) for doc in corpus_cleaned]
    print('Took {:.0f} minutes'.format((time.time() - t0)/60))

    df['Content_clean'] = corpus_tokenized

    cols = df.columns.tolist()
    cols2 = cols[:3] + cols[-1:] + cols[3:-1]
    df = df[cols2]
    
    filename = 'data/8ks_with_returns_cleaned_v2.csv'
    df.to_csv(filename)
    print('Saved {}'.format(filename))