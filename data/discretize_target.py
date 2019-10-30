# -*- coding: utf-8 -*-

df = pd.read_csv('data/8ks_with_returns_cleaned.csv', parse_dates=[1])
    
# extract list of documents (strings)
corpus = df.Content.values.tolist()
d = df['1-day'].values

y_up = 2*(d >= 0.01).astype(int)
y_mid = 1*((d<0.01)&(d>-0.01)).astype(int)
y_down = 0*(d <= -0.01).astype(int)

y = y_up + y_mid + y_down

y_up = (df['1-day'] > 0.01).astype(int).values
y_mid = (df['1-day'] > 0.01).astype(int).values
y_down = (df['1-day'] > 0.01).astype(int).values
    