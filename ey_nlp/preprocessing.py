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

#%%
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
    Clean then tockenize data
    """
    doc_clean = preprocess_text(doc)
    doc_clean = custom_tokenizer(doc_clean)
    
    doc_as_string = ' '.join(doc_clean)
    return doc_as_string

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
def discretize_target(df):
    """Turn target into 3 classes: -.01 to 0.01, above, and below
    """
    
    df1 = df.copy()
    
    for col in ['1-day', '2-day', '3-day', '5-day', '10-day', '20-day', '30-day']:
        d = df[col].fillna(0).values
        
        new_column_name = 'ret_' + col
        
        y_up = 2*(d >= 0.01).astype(int)
        y_mid = 1*((d<0.01)&(d>-0.01)).astype(int)
        y_down = 0*(d <= -0.01).astype(int)
        
        y = y_up + y_mid + y_down
        
        df1.loc[:, new_column_name] = y

    return df1

#%%
if __name__ == "__main__":
    df = pd.read_csv('data/8ks_with_returns.csv', parse_dates=['Date'], index_col=False)
    
    # add sentiment
    df1 = pd.read_csv('data/8ks_return_sentiment.csv', parse_dates=['Date'], index_col=False)
    df2 = df1.loc[:,['Ticker', 'Date', 'sentiment']].drop_duplicates(subset=['Ticker','Date'])
    df3 = df.merge(df2, how='left', on=['Ticker', 'Date'])
    
    del df
    df = df3
    
    # get text
    corpus = df.Content.values.tolist()
    
    # clean text
    print('Cleaning text... \nThis could take 30 minutes')
    
    t0 = time.time()
    corpus_cleaned = [clean_text(doc) for doc in corpus]
    print('Done in {:.0f} minutes'.format((time.time() - t0)/60))
    
    df.loc[:,'Content_clean'] = corpus_cleaned
    df.loc[:,'Content_clean'] = df.loc[:,'Content_clean'].fillna('')

    # to put cleaned content next to original
    cols = ['Ticker', 'Date', 'Content', 'Content_clean', 'Close', '1-day', '2-day', '3-day', '5-day', '10-day', '20-day', '30-day', 'sentiment']
    df = df[cols]
   
    df1 = discretize_target(df)
    
    # split to train and test
    df2 = df1.sort_values(by=['Date','Ticker']).set_index('Date')   
    df2.loc[:,'month'] = df2.index.month
    df2.loc[:,'year'] = df2.index.year
    
    cutoff_date = pd.Timestamp(year=2018, month=9, day=1, hour=0)
    print('Train-test split on {}'.format(cutoff_date))
    train = df2.loc[df2.index <= cutoff_date].reset_index().drop(columns=['month', 'year'])
    test = df2.loc[df2.index > cutoff_date].reset_index().drop(columns=['month', 'year'])
    
    # save
    filename = 'data/train.csv'
    train.to_csv(filename, index=False)
    print('Saved {}'.format(filename))
    
    filename = 'data/test.csv'
    test.to_csv(filename, index=False)
    print('Saved {}'.format(filename))