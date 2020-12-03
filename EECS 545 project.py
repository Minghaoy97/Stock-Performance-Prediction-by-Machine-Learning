#!/usr/bin/env python
# coding: utf-8

# In[60]:


import xlrd  

print('Please input the fundamental xlsx file name:') # For example, Coke.xlsx
fundamental_filename = input()
print('Please input the price cvs file name:')# For example, COKE.csv
price_filename = input()


# In[61]:


import pandas as pd
from collections import defaultdict
from pandas.core.frame import DataFrame

fund_file = xlrd.open_workbook(fundamental_filename)

datasheet = fund_file.sheets()[0]
select_data = []
for i in range(datasheet.nrows):  
    if '' not in datasheet.row_values(i):
        select_data.append(datasheet.row_values(i))
        
for i in range(len(select_data)):
    for j in range(len(select_data[i])):
        if select_data[i][j] == '-':
            select_data[i][j] = select_data[i][j - 1]

fund_data = defaultdict(list)
for i in select_data:
    fund_data[i[0]] = i[1:]
data = DataFrame(fund_data)


# In[62]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

def gen_simple_ma(df, ma = 5):
    df["MA" + str(ma)] = df["Close"].rolling(window=ma, center=False).mean()
    return df

def gen_ewma(df, ma = 5):
    df["EMA" + str(ma)] = df["Close"].ewm(span = ma, min_periods = ma).mean()
    return df

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

    for j in range(periods, length):
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
        rsies[j] = 100 - 100 / (1 + rs)
    df["RSI"+" "+str(periods)] = rsies
    return df

def gen_ADO(data, trend_periods=5, open_col='Open', high_col='High', low_col='Low', close_col='Close', vol_col='Volume'):
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            ac = ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            ac = 0
        data.at[index, 'ADO'] = ac
#     data['acc_dist_ema' + str(trend_periods)] = data['acc_dist'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    return data

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

def gen_Momentum(data,trend_periods=9,column='Close'):
    array_list = data["Close"]
    length = len(array_list)
    Momen=[]
    for i in range(trend_periods,length):
        M=array_list[i]-array_list[i-trend_periods]
        Momen.append(M)
    Momen=[np.nan]*(trend_periods)+Momen
    data["Momentum" + str(trend_periods)]  =  Momen
    return data

def gen_Larry(data,periods_long=14,open_col='Open', high_col='High', low_col='Low', close_col='Close'):
    Close = data[close_col]
    Open = data[open_col]
    High = data[high_col]
    Low = data[low_col]
    Larry=[]
    length=len(Close)
    for i in range(periods_long,length):
        Hn=High[i:i+periods_long].max()
        Ln=Low[i:i+periods_long].min()
        L=(Hn-Close[i])*100/(Hn-Ln)
        Larry.append(L)
    Larry=[np.nan]*periods_long+Larry
    data["Larry" + str(periods_long)]  =  Larry
    return data

def gen_KandD_percent(df, n=9):
    low_list = df['Low'].rolling(9, min_periods=9).min()
    low_list.fillna(value = df['Low'].expanding().min(), inplace = True)
    high_list = df['High'].rolling(9, min_periods=9).max()
    high_list.fillna(value = df['High'].expanding().max(), inplace = True)
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    df['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

def gen_CCI(data, n, high_col='High', low_col='Low', close_col='Close'):
    M = np.array((data[high_col] + data[low_col] + data[close_col])/3)
    N = np.ones(n)
    w = N/n
    SM = np.convolve(w,M)[n-1:-n+1]
    SM = np.array(SM)
    SM = np.r_[np.zeros(n-1),SM]
    D = []
    for i in range (0,n):
        D.append(0)
    for i in range (n,M.shape[0]):
        d = 0
        for j in range (0,n):
            e = abs(M[i-j] - SM[i])
            d = d + e
        d = d/n
        D.append(d)
    D = np.array(D)
    D = D + 0.001
    CCI = (M - SM)/D/0.015
    data["CCI" + str(n)]  =  CCI
    return data


# In[63]:


df = pd.read_csv(price_filename)

df = gen_simple_ma(df,ma = 5)
df = gen_ewma(df, ma = 5)
df = gen_ADO(df, trend_periods=5)
df = gen_macd(df, period_long=26, period_short=12, period_signal=9)
df = gen_average_true_range(df, trend_periods=14)
df = gen_Momentum(df,trend_periods=9)
df = gen_Larry(df,periods_long=14)
df = gen_KandD_percent(df, 5)
df = gen_CCI(df, 5)


# In[64]:


complete_data = DataFrame()
month_label = ['9', '6', '3', '12']
season_label = 0 # point to the according season fundamental data
year_label = 0
month_index = -1
combined_df = pd.DataFrame()

for i in range(len(df)):
    date = df.loc[i][0].split('-') #list of date
    year_label = int(date[0])
    month = int(date[1])
    if month > 9 and month <= 12:
        month_index = 3
        season_label = 3 + 4 * (2019 - year_label)
    elif month > 0 and month <= 3:
        month_index = 2
        season_label = 2 + 4 * (2020 - year_label)
    elif month > 3 and month <= 6:
        month_index = 1
        season_label = 1 + 4 * (2020 - year_label)
    else:
        month_index = 0
        season_label = 4 * (2020 - year_label)
    tmp_df = pd.DataFrame(df.loc[i].append(data.loc[season_label])).T
    combined_df = combined_df.append(tmp_df)


# In[65]:


combined_file_name = price_filename.split('.')[0] + '_combined.csv'
combined_df.to_csv(combined_file_name, index = False)

