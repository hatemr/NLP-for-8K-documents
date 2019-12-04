# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:42:50 2019

@author: Max

Backtest NLP model on 8ks and show the results

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
#from tools import lp, sp, beep
from IPython import embed

# Change path to where the 'predictions' file is and run the script
#path = 'D:/Data/nlp/'
#with open(path+'oos_preds.pickle', 'rb') as file:
#    data = pickle.load(file)



def lp(name):
    filename = name + ".pickle"
    with open(filename, 'rb') as handle:
        inter = pickle.load(handle)
    return inter

def ternary_determine(tup):
    """ Return 'Buy'/'Sell'/'Hold' from a tuple of probabilities.
        Assumes that the first value (y_pred_0) corresponds to 'Sell'. """
    choice = max(tup)
    if tup[0] == choice:
        return ('Sell',choice)
    elif tup[2] == choice:
        return ('Buy',choice)
    else:
        return ('Hold',choice)

def trade(data):
    """ Filter only 'Buy' or 'Sell' signals and reverse the Sells to account for shorting """
    n_day = 'ret_1-day'
    returns = data[['Date', n_day, 'y_pred_0', 'y_pred_1', 'y_pred_2']].copy()
    returns['tup'] = list(map(ternary_determine, np.array(returns[['y_pred_0', 'y_pred_1', 'y_pred_2']])))
    returns['trade'] = [x[0] for x in returns['tup']]
    returns['conf'] = [x[1] for x in returns['tup']]
    buys = returns[returns['trade'] == 'Buy'].copy()
    sells = returns[returns['trade'] == 'Sell'].copy()
    sells.loc[:,n_day]*=-1  # reverse the losses when we short these stocks
    trades = pd.concat([buys, sells], axis=0).sort_values('Date')
    trades.set_index('Date', inplace=True)
    trades.index = pd.to_datetime(trades.index)
    return trades.drop(columns=['y_pred_0', 'y_pred_1', 'y_pred_2', 'tup', 'trade'])

def full_strat(trades):
    """ Allocate entire current portfolio equally among each day's trades """
    account = [100]
    dex = [trades.index[0]-pd.Timedelta(days=1)]
    for day in pd.date_range(trades.index[0],trades.index[-1]):
        if day in sorted(list(set(trades.index))):
            totalret = float(np.mean(trades.loc[day]['1-day']))
            account.append(account[-1]*(1+totalret))
            dex.append(day)
        else:
            account.append(account[-1])
            dex.append(day)
    return pd.Series(account, index=dex)/100

def smart_strat(trades):
    """ Allocate entire current portfolio equally among each day's trades """
    account = [100]
    dex = [trades.index[0]-pd.Timedelta(days=1)]
    for day in pd.date_range(trades.index[0],trades.index[-1]):
        if day in sorted(list(set(trades.index))):
            # for each day, divide current account value equally and add each trade's returns
            daytrades = trades.loc[day]
            if type(daytrades['conf']) == pd.Series:
                total_return = float((daytrades['ret_1-day'].dot(daytrades['conf']))/(sum(daytrades['conf'])*len(daytrades)))
            else:
                total_return = daytrades['ret_1-day']
            account.append((1+total_return) * account[-1])
            dex.append(day)
        else:
            account.append(account[-1])
            dex.append(day)
    return pd.Series(account, index=dex)/100

def main(data):
    """ Show the results of trading strategys. """
    trades = trade(data)
    trades.index = pd.to_datetime(trades.index)
    trades.dropna(inplace=True)
    #fullstrat = full_strat(trades)
    smart = smart_strat(trades)
#    plt.plot(fullstrat)
    plt.plot(smart)
    plt.title('Strategy Returns')
#    plt.legend(['FullAlloc Strat', 'SmartAlloc Strat'])
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.gcf().autofmt_xdate()
    return
#main()

def test(grid):
    testdata = lp('data/test_bert')
    testdata['sentiment_x'] = testdata['sentiment_x'].fillna(0)
    data = pd.DataFrame(grid.predict_proba(testdata.drop(columns='ret_1-day')))
    data.columns = ['y_pred_0','y_pred_1','y_pred_2']
    data = pd.merge(testdata,data,how='left',left_index=True,right_index=True)
    main(data)
    return

def compare():
    for x in ['4']:#,'6','lda1','bert1']:#['2','3','4','5','6','rf1','lda1']:
        grid = lp('models/grid_search_ret_1-day_bert_v1')
        test(grid)
    plt.legend(['bert1'])
    return





















