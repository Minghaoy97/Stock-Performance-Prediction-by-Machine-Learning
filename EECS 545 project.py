#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd


# In[112]:


df = pd.read_csv("D:/UMICH/EECS 545/project/AAPL.csv")
# df.set_index("Date", inplace=True)


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


# In[105]:


def gen_macd(data, period_long=26, period_short=12, period_signal=9, column='Close'):
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


# In[116]:


def gen_average_true_range(data, trend_periods=14, open_col='Open', high_col='High', low_col='Low', close_col='Close', drop_tr = True):
    for index, row in data.iterrows():
        prices = [row[high_col], row[low_col], row[close_col], row[open_col]]
        if index > 0:
            val1 = np.amax(prices) - np.amin(prices)
            val2 = abs(np.amax(prices) - data.at[index - 1, close_col])
            val3 = abs(np.amin(prices) - data.at[index - 1, close_col])
            true_range = np.amax([val1, val2, val3])

        else:
            true_range = np.amax(prices) - np.amin(prices)

        data.at[index, 'true_range'] = true_range
    data['ATR'] = data['true_range'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    if drop_tr:
        data = data.drop(['true_range'], axis=1)
        
    return data


# In[ ]:
def Momentum(data,trend_periods=9,column='Close'):
    array_list = data["Close"]
    length = len(array_list)
    Momen=[]
    for i in range(10,length):
        M=array_list[i]-array_list[i-trend_periods]
        Momen.append(M)
    Momen=[np.nan]*10+Momen
    data["Momentum" + str(trend_periods)]  =  Momen
    return data

# In[ ]:
def Larry(data,periods_long=14,open_col='Open', high_col='High', low_col='Low', close_col='Close'):
    Close = data[close_col]
    Open = data[open_col]
    High = data[high_col]
    Low = data[low_col]
    Larry=[]
    length=len(Close)
    for i in range(periods_long+1,length):
        Hn=High[i:i+periods_long].max()
        Ln=Low[i:i+periods_long].min()
        L=(Hn-Close[i])*100/(Hn-Ln)
        Larry.append(L)
    Larry=[np.nan]*15+Larry
    data["Larry" + str(periods_long)]  =  Larry
    return data

def gen_KandD_percent(data, n=5):
    for j in range(len(data)):
        if j - (n-1) <= 0:
            data.at[j,"Stochastic_K%"] = 0
        else:
            LL = data[j-n+1:j+1]["Low"].min()
            HH = data[j-n+1:j+1]["High"].max()
            data.at[j,"Stochastic_K%"] = ((data.iloc[j]["Close"] - LL)/(HH - LL)) * 100
    for j in range(len(data)):
        if j - (n-1) <= 0:
            data.at[j,"Stochastic_D%"] = 0
        else:
            value = 0
            for i in range(n):
                value += data.at[j-(n - i),"Stochastic_K%"]
            data.at[j,"Stochastic_D%"] = value / 1000
    return data

def gen_CCI(data, n, high_col='High', low_col='Low', close_col='Close'):
    M = (high_col + low_col + close_col)/3
    N = np.ones(n)
    w = N/n
    SM = np.convolve(w,M)[n-1:-n+1]
    D = []
    for i in range (0,M.shape[0])
        d = 0
        for j in range (0,n)
            e = abs(M[i-j] - SM[i])
            d = d + e
        d = d/n
        D = D.append(d)
    CCI = (M - SM)/(0.015*D)
    data["CCI" + str(n)]  =  CCI
    return data
