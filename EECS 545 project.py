#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd


# In[5]:


df = pd.read_csv("D:/UMICH/EECS 545/project/AAPL.csv")
df.set_index("Date", inplace=True)


# In[27]:


def gen_simple_ma(df, ma = 5):
    df["MA" + str(ma)] = df["Close"].rolling(window=ma, center=False).mean()
    return df


# In[53]:


def gen_ewma(df, ma = 5):
    df["EMA" + str(ma)] = df["Close"].ewm(span = ma, min_periods = ma).mean()
    return df


# In[66]:


def gen_RSI(df, periods=5):
    array_list = df["Close"]
    length = len(array_list)
    rsies = [np.nan] * length
    if length <= periods:
        return rsies
    up_avg = 0
    down_avg = 0

    first_t = array_list[:periods + 1]
    for i in range(1, len(first_t)):
        if first_t[i] >= first_t[i - 1]:
            up_avg += first_t[i] - first_t[i - 1]
        else:
            down_avg += first_t[i - 1] - first_t[i]
    up_avg = up_avg / periods
    down_avg = down_avg / periods
    rs = up_avg / down_avg
    rsies[periods-1] = 100 - 100 / (1 + rs)

    for j in range(periods + 1, length):
        up = 0
        down = 0
        if array_list[j] >= array_list[j - 1]:
            up = array_list[j] - array_list[j - 1]
            down = 0
        else:
            up = 0
            down = array_list[j - 1] - array_list[j]
        up_avg = (up_avg * (periods - 1) + up) / periods
        down_avg = (down_avg * (periods - 1) + down) / periods
        rs = up_avg / down_avg
        rsies[j-1] = 100 - 100 / (1 + rs)
    return rsies


# In[93]:


def gen_ADO(data, trend_periods=5, open_col='Open', high_col='High', low_col='Low', close_col='Close', vol_col='Volume'):
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            ac = ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            ac = 0
        data.at[index, 'ADO'] = ac
#     data['acc_dist_ema' + str(trend_periods)] = data['acc_dist'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    return data


# In[102]:


def macd(data, period_long=26, period_short=12, period_signal=9, column='Close'):
    remove_cols = []
    if not 'EMA' + str(period_long) in data.columns:
        data = gen_ewma(data, period_long)
        remove_cols.append('EMA' + str(period_long))

    if not 'EMA' + str(period_short) in data.columns:
        data = gen_ewma(data, period_short)
        remove_cols.append('EMA' + str(period_short))

    data['macd_val'] = data['EMA' + str(period_short)] - data['EMA' + str(period_long)]
    data['macd_signal_line'] = data['macd_val'].ewm(ignore_na=False, min_periods=0, com=period_signal, adjust=True).mean()

    data = data.drop(remove_cols, axis=1)
        
    return data


# In[ ]:




