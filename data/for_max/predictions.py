# -*- coding: utf-8 -*-

# class predictions from random forest cross-validated on grid search with LDA

data = pd.read_csv('data.csv', parse_dates=['Date'])

# predictions (multiclass) are in last three columns

# 0: < -0..01
# 1: between -0.01 and 0.01
# 2: > 0.01

data2.head()