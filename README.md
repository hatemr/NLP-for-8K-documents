# EY-NLP
Predicting returns from 8K documents using text analysis.

## Steps
1. Download 8K documents from today's S&P 500 companies for the past 5 years.
2. Extract the useful text from the html documents.
3. Transform the raw text into useful features.
  1. remove proper nouns
  2. normalization
    * change all words to lower case.
    * expand contractions such as "haven't" to "have not".
    * delete numbers, punctuations, special symbols, and non-English words.
4. Predict forward returns over several horizons.

## Implement from paper
I replicated the paper. Hopefully it works.
