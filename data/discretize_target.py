# -*- coding: utf-8 -*-


'''
Discretizes the returns and saves as data.csv

'''
import pandas as pd

df = pd.read_csv('data/8ks_with_returns_cleaned.csv', parse_dates=[1])
    
#%%

for col in ['1-day', '2-day', '3-day', '5-day', '10-day', '20-day', '30-day']:
    d = df[col].fillna(0).values
    
    new_column_name = 'ret_' + col
    # extract list of documents (strings)
    print(new_column_name)
    
    y_up = 2*(d >= 0.01).astype(int)
    y_mid = 1*((d<0.01)&(d>-0.01)).astype(int)
    y_down = 0*(d <= -0.01).astype(int)
    
    y = y_up + y_mid + y_down
    
    df[new_column_name] = y

#%%    
df1 = df.drop(columns=['Unnamed: 0'])

df1.to_csv('data/data.csv', index=False)

#%%
df2 = pd.read_csv('data/data.csv', parse_dates=['Date'])