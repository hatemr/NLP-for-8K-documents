# NLP Strategy Overview
Predicting returns from 8K documents using text analysis and natural language processing.

![strategy](images/strategy_overview.png)

## Data Preparation
1. Download 8K documents from today's S&P 500 companies for the past 5 years.
2. Extract the useful text from the html documents.
3. Clean text.
  * remove proper nouns (Apple)
  * make lower case (The -> the)
  * expand contractions (can't -> cannot)
  * remove special characters and digits ([^a-zA-z0-9\s])
  * remove stopwords (a, the)
  * remove html tags (<p></p>)
  * remove accented characters
  * remove newlines ([\r|\n|\r\n]+)
  * remove extra whitespace
4. Tokenize text
  * lemmatize
  * stemmer
5. Vectorize to a document-term matrix using `CountVectorizer`.
  * All our models require creating the document-term matrix. However, we 
  might later try models that use another vectorizer (e.g. tf-idf).
  
## Modeling
The key step in our strategy is to turn the document-term matrix into a
lower-dimensional representation.

## Implement from paper
I replicated the paper. Need data to estimate on. Also, need a strategy to use this sentiment factor.

## LDA for Topic Modeling
Latent Dirichlet Allocation (LDA) aims to model documents as arising from multiple topics, where a _topic_ is defined to be a distribution over a fixed vocabulary of terms. Each document exhibits these topics with different proportions. The K topics and their relative weights are treated as hidden variables. Given a collection of documents, the _posterior distribution_ of the hidden variables given the observed documents determines a hidden topical decomposition of the collection. 

### Resources
* [TOPIC MODELS](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.186.4283&rep=rep1&type=pdf)
* [Hierarchical Dirichlet Processes](https://www.stat.berkeley.edu/~aldous/206-Exch/Papers/hierarchical_dirichlet.pdf)