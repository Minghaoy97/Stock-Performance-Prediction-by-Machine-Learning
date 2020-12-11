#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
import sklearn
from sklearn import neural_network
from sklearn.metrics import mean_squared_error
import time


# In[104]:


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

# In[ ]:
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

def gen_CCI(data, n=5, high_col='High', low_col='Low', close_col='Close'):
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
    return(data)

def get_all_data(data):
    data = gen_ADO(data)
    data = gen_average_true_range(data)
    data = gen_CCI(data)
    data = gen_ewma(data)
    data = gen_KandD_percent(data)
    data = gen_Larry(data)
    data = gen_macd(data)
    data = gen_Momentum(data)
    data = gen_RSI(data)
    data = gen_simple_ma(data)
    return data

def gen_pair(path, shift_day=-1, train_num = 1000):
    df = pd.read_csv(path)
    df = get_all_data(df)
    df.dropna(inplace=True)
#     x = df[["Open","High","Low","Volume","ADO","ATR","CCI5","EMA5","K","D","J","Larry14","macd_val","macd_signal_line","Momentum9","RSI 5","MA5"]]
    x = df.loc[:,"MA5":]
    y = np.array(df["Close"]).reshape(-1,1)
#     returns = df["Close"].shift(-1)/df["Close"]-1
#     returns[returns>0] = 1
#     returns[returns<0] = 0
#     y = np.array(returns).reshape(-1,1)
    y = pd.DataFrame(y)
    y = np.array(y.shift(shift_day)).reshape(-1,1)
    y = y[:shift_day]
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    x = np.array(x).reshape(-1,x.shape[1])[:shift_day]
    x_train = x[:train_num]
    y_train = y[:train_num]
    x_test = x[train_num:]
    y_test = y[train_num:]
    return df,x,y,x_train,y_train,x_test,y_test


# In[3]:


stock_list = ["AAPL","COKE","HP","KR","MCD","MSFT","RCL"]
stock_combined_list = []
for stock in stock_list:
    stock_combined_list.append(stock+"_combined")


# In[4]:


c = [2**(-1),2,2**(3),2**(5),2**(7),2**(9)]
d = [1,2,3]
kernel_function = ["linear", "poly", "rbf"]
shift_list = [-1,-5,-10]
def gen_mse_dic(kernel_num = 0):
    dic = dict()
    for i in range(len(c)):
        dic[c[i]] = {}
        for j in range(len(stock_list)):
            stock = stock_list[j]
            df,x,y,x_train,y_train,x_test,y_test = gen_pair(f"D:/UMICH/EECS 545/project/{stock}.csv", shift_day=shift_list[0],train_num=4500)
            model = svm.SVR(kernel = kernel_function[kernel_num],C=c[i])
            model.fit(x_train,y_train)
            y_predicted = model.predict(x_test)
            MSE = np.sqrt(mean_squared_error(y_predicted,y_test))
            dic[c[i]][stock_list[j]] = MSE
        st = c[i]
        print(f"{st} done!")
    return dic


# In[5]:


def gen_mse_poly(kernel_num = 1):
    dic = dict()
    for i in range(len(c)):
        dic[c[i]] = {}
        for z in range(len(d)):
            dic[c[i]][d[z]] = {}
            for j in range(len(stock_list)):
                stock = stock_list[j]
                df,x,y,x_train,y_train,x_test,y_test = gen_pair(f"D:/UMICH/EECS 545/project/{stock}.csv", shift_day=shift_list[0],train_num=4500)
                model = svm.SVR(kernel = kernel_function[kernel_num],C=c[i],degree=d[z])
                model.fit(x_train,y_train)
                y_predicted = model.predict(x_test)
                MSE = np.sqrt(mean_squared_error(y_predicted,y_test))
                dic[c[i]][d[z]][stock_list[j]] = MSE
        st = c[i]
        print(f"{st} done!")
    return dic


# In[6]:


dict_poly = {0.5: {1: {'AAPL': 2.4179752329815285,
   'COKE': 11.267202701383626,
   'HP': 2.062593345954115,
   'KR': 0.8107014674683446,
   'MCD': 4.722919891357044,
   'MSFT': 4.1039499251503155,
   'RCL': 4.722804609960944},
  2: {'AAPL': 211.11185294386516,
   'COKE': 166.55646118150935,
   'HP': 19.412262292447334,
   'KR': 10.0203154662747,
   'MCD': 311.8398152523901,
   'MSFT': 141.9337159763402,
   'RCL': 16.13776090558803},
  3: {'AAPL': 628.6810434415065,
   'COKE': 638.3118124084107,
   'HP': 10.706868580442011,
   'KR': 9.643110066214437,
   'MCD': 225.63938754207828,
   'MSFT': 986.0207153085969,
   'RCL': 27.801621475170787}},
 2: {1: {'AAPL': 2.415784007610637,
   'COKE': 11.272821509958975,
   'HP': 2.06354470418371,
   'KR': 0.8108194248271929,
   'MCD': 4.727488125398318,
   'MSFT': 4.1027608195645895,
   'RCL': 4.7283563423809145},
  2: {'AAPL': 211.3755464997476,
   'COKE': 170.26353172657923,
   'HP': 19.415734458200482,
   'KR': 10.020959006983931,
   'MCD': 313.01680757803143,
   'MSFT': 158.20611153827963,
   'RCL': 16.115128655384623},
  3: {'AAPL': 628.9398941053369,
   'COKE': 641.1083215228894,
   'HP': 10.704064713083438,
   'KR': 9.635633814874266,
   'MCD': 225.76457295513688,
   'MSFT': 990.3433492922129,
   'RCL': 27.89190157630733}},
 8: {1: {'AAPL': 2.41523105400267,
   'COKE': 11.274545880230805,
   'HP': 2.063932557850565,
   'KR': 0.8108317370639357,
   'MCD': 4.728722445977445,
   'MSFT': 4.1025857261943335,
   'RCL': 4.729566662522425},
  2: {'AAPL': 211.45147972343585,
   'COKE': 170.73454704468398,
   'HP': 19.415731898131604,
   'KR': 10.019266570746272,
   'MCD': 313.237853957057,
   'MSFT': 158.8954153310651,
   'RCL': 16.116464902175746},
  3: {'AAPL': 628.9430813576005,
   'COKE': 641.1113807234141,
   'HP': 10.70120786721719,
   'KR': 9.635652042588497,
   'MCD': 225.76469953262398,
   'MSFT': 990.3646971151863,
   'RCL': 27.891774047968276}},
 32: {1: {'AAPL': 2.4153214160180894,
   'COKE': 11.274869054593001,
   'HP': 2.0638681848312217,
   'KR': 0.810834807379426,
   'MCD': 4.7291103950871305,
   'MSFT': 4.102583888306432,
   'RCL': 4.7303973955106535},
  2: {'AAPL': 211.44974110240148,
   'COKE': 171.75136561915323,
   'HP': 19.415721658005463,
   'KR': 10.019236468818683,
   'MCD': 313.2827963240403,
   'MSFT': 158.89149456769294,
   'RCL': 16.116717261825},
  3: {'AAPL': 628.9475599454512,
   'COKE': 641.6840745382825,
   'HP': 10.701278196697418,
   'KR': 9.635724954486296,
   'MCD': 225.76520584268548,
   'MSFT': 990.4500884268313,
   'RCL': 27.891263941934238}},
 128: {1: {'AAPL': 2.4152054221247448,
   'COKE': 11.275059788200956,
   'HP': 2.063867117188662,
   'KR': 0.8108327908623377,
   'MCD': 4.729131394587549,
   'MSFT': 4.102577529438008,
   'RCL': 4.73036961360869},
  2: {'AAPL': 211.45097378720223,
   'COKE': 171.75224758558056,
   'HP': 19.415680697301852,
   'KR': 10.0192135077822,
   'MCD': 313.36296614133886,
   'MSFT': 158.8758116580505,
   'RCL': 16.11582888585462},
  3: {'AAPL': 628.9481894250356,
   'COKE': 642.5239768351631,
   'HP': 10.70109467873127,
   'KR': 9.636016640296933,
   'MCD': 226.02821851038564,
   'MSFT': 990.7947238921582,
   'RCL': 27.889223615680617}},
 512: {1: {'AAPL': 2.4150896400526096,
   'COKE': 11.275038139406796,
   'HP': 2.063821489076982,
   'KR': 0.8108158205175583,
   'MCD': 4.729125768057715,
   'MSFT': 4.10254700071018,
   'RCL': 4.730459309524733},
  2: {'AAPL': 211.43689218188626,
   'COKE': 171.75577546222516,
   'HP': 19.415516854841513,
   'KR': 10.018635393343184,
   'MCD': 313.37180429732683,
   'MSFT': 158.81308099113915,
   'RCL': 16.112277907214867},
  3: {'AAPL': 629.1170395407838,
   'COKE': 642.9482217051664,
   'HP': 10.700732495145521,
   'KR': 9.636974756662804,
   'MCD': 226.05892933833093,
   'MSFT': 991.8878702305935,
   'RCL': 27.881063883959342}}}


# In[7]:


reform_poly = {(f"C = {level1_key}", f"degree = {level2_key}", level3_key): [values]
          for level1_key, level2_dict in dict_poly.items()
          for level2_key, level3_dict in level2_dict.items()
          for level3_key, values      in level3_dict.items()}
df_poly = pd.DataFrame(reform_poly).T
df_poly.to_csv("df_poly.csv")


# In[8]:


dict_linear = {0.5: {'AAPL': 2.42187550916986,
  'COKE': 11.264300679377932,
  'HP': 2.0626618668248,
  'KR': 0.8107002908352684,
  'MCD': 4.720448848079527,
  'MSFT': 4.119995687960477,
  'RCL': 4.722094223213366},
 2: {'AAPL': 2.4160159799379537,
  'COKE': 11.270642432711899,
  'HP': 2.0637617344636316,
  'KR': 0.8107692633660293,
  'MCD': 4.726384483213756,
  'MSFT': 4.104012448010299,
  'RCL': 4.7275607534598745},
 8: {'AAPL': 2.4154014520741938,
  'COKE': 11.274415368129054,
  'HP': 2.063921968715703,
  'KR': 0.8108245133125832,
  'MCD': 4.728341632665992,
  'MSFT': 4.102778797979348,
  'RCL': 4.729602034217651},
 32: {'AAPL': 2.415274643788004,
  'COKE': 11.274876037636757,
  'HP': 2.0639359617573674,
  'KR': 0.8108338100417565,
  'MCD': 4.729183536145425,
  'MSFT': 4.102585801205743,
  'RCL': 4.729969968379239},
 128: {'AAPL': 2.4152363770167886,
  'COKE': 11.27486188863409,
  'HP': 2.0639122403714936,
  'KR': 0.8108351411514393,
  'MCD': 4.72918419559849,
  'MSFT': 4.10258417223984,
  'RCL': 4.730378207452415},
 512: {'AAPL': 2.415214256650152,
  'COKE': 11.274966648107936,
  'HP': 2.063967752572758,
  'KR': 0.8108493225779684,
  'MCD': 4.729217432825456,
  'MSFT': 4.102577892208263,
  'RCL': 4.730239504051308}}


# In[9]:


reform_linear = {(f"C = {key1}",key2): [values]
for key1,dict2 in dict_linear.items()
for key2, values in dict2.items()}
df_linear = pd.DataFrame(reform_linear).T
df_linear.to_csv("df_linear.csv")


# In[10]:


dict_rbf = {0.5: {'AAPL': 40.27118563392452,
  'COKE': 120.51159830346552,
  'HP': 2.081540973501589,
  'KR': 0.8164407906427679,
  'MCD': 65.50834760190035,
  'MSFT': 90.67071904665984,
  'RCL': 5.659181102249438},
 2: {'AAPL': 39.089750266147334,
  'COKE': 107.02978908524113,
  'HP': 2.076614033173245,
  'KR': 0.8112002706286621,
  'MCD': 48.74753806324427,
  'MSFT': 89.32194057094344,
  'RCL': 4.9714498930167155},
 8: {'AAPL': 38.579610952038095,
  'COKE': 97.62897793987646,
  'HP': 2.076135598791987,
  'KR': 0.8115904562847599,
  'MCD': 35.7457999307169,
  'MSFT': 87.41404171532338,
  'RCL': 4.776542360415253},
 32: {'AAPL': 38.81214589169046,
  'COKE': 88.7686301448567,
  'HP': 2.0773522755617426,
  'KR': 0.8128855286625214,
  'MCD': 28.169235895254392,
  'MSFT': 86.75712633432722,
  'RCL': 4.756289286920397},
 128: {'AAPL': 39.20124542096546,
  'COKE': 83.99249408448665,
  'HP': 2.0776221512741357,
  'KR': 0.8142434964905405,
  'MCD': 27.702209676111792,
  'MSFT': 84.24134173063797,
  'RCL': 4.749262743815492},
 512: {'AAPL': 40.27468430724524,
  'COKE': 84.84654525883207,
  'HP': 2.0773095301181375,
  'KR': 0.8151887945831868,
  'MCD': 30.95445637003738,
  'MSFT': 82.24405746384991,
  'RCL': 4.746857379266873}}


# In[11]:


reform_rbf = {(f"C = {key1}",key2): [values]
for key1,dict2 in dict_rbf.items()
for key2, values in dict2.items()}
df_rbf = pd.DataFrame(reform_rbf).T
df_rbf.to_csv("df_rbf.csv")


# In[121]:


df[4500:]["Date"].values


# In[112]:


l = []
for i in range(len(df[4500:]["Date"].values)):
    l.append(time.mktime(time.strptime(df[4500:]["Date"].values[i], "%Y/%m/%d")))
l


# In[132]:


pd.date_range(start='2018-11-28 08:10:50',periods=11,freq='M',normalize=True)


# In[147]:


shift_day = -1
stock = stock_list[2]
df,x,y,x_train,y_train,x_test,y_test = gen_pair(f"D:/UMICH/EECS 545/project/{stock}.csv", shift_day=shift_day,train_num=4500)
model = svm.SVR(kernel="linear", C = 0.5)
model.fit(x_train,y_train)
print(np.sqrt(mean_squared_error(model.predict(x_test), y_test)))
print(model)
plt.figure(figsize=(15,7))
plt.plot(y_test, label = "Actual",color="blue")
plt.plot(model.predict(x_test), "r",label="Predicted")
plt.legend()
plt.title(f"Predict HP for {-shift_day} day")
l = []
for i in df[4500:]["Date"].values:
    l.append(i[:-3])
ticks = sorted(list(set(l)))
plt.xticks(np.arange(0,len(l)+1,20),ticks, rotation=60)
plt.ylabel('Stock Price',fontsize=14)
# plt.xlim((0,462))


# In[13]:


print(len(x),len(y),len(x_train),len(y_train),len(x_test),len(y_test),len(model.predict(x_test)))


# In[ ]:




