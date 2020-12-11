import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier as RF
import graphviz




def RF_regression(filename,t0,t1,point=(0.98,0.99,1,1.01,1.02),k=1,d=0.01,n=5,generateindices=False,output=False,steplabel=True):
    df = pd.read_csv(filename)

    # Seperate Date
    df['Y'], df['M'], df['Da'] = df["Date"].str.split('/').str
    df['Y'].astype("int")
    df['M'].astype("int")
    df['Da'].astype("int")
    del df['Date']

    #generate tech indicies
    if generateindices==True:
        def gen_simple_ma(df, ma=5):
            df["MA" + str(ma)] = df["Close"].rolling(window=ma, center=False).mean()
            return df

        def gen_ewma(df, ma=5):
            df["EMA" + str(ma)] = df["Close"].ewm(span=ma, min_periods=ma).mean()
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
            rsies[periods - 1] = 100 - 100 / (1 + rs)

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
            df["RSI" + " " + str(periods)] = rsies
            return df

        def gen_ADO(data, trend_periods=5, open_col='Open', high_col='High', low_col='Low', close_col='Close',
                    vol_col='Volume'):
            for index, row in data.iterrows():
                if row[high_col] != row[low_col]:
                    ac = ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (
                                row[high_col] - row[low_col]) * \
                         row[vol_col]
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
            data['macd_signal_line'] = data['macd_val'].ewm(ignore_na=False, min_periods=0, com=period_signal,
                                                            adjust=True).mean()

            data = data.drop(remove_cols, axis=1)

            return data

        def gen_average_true_range(data, trend_periods=14, open_col='Open', high_col='High', low_col='Low',
                                   close_col='Close',
                                   drop_tr=True):
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

        def gen_Momentum(data, trend_periods=9, column='Close'):
            array_list = data["Close"]
            length = len(array_list)
            Momen = []
            for i in range(trend_periods, length):
                M = array_list[i] - array_list[i - trend_periods]
                Momen.append(M)
            Momen = [np.nan] * (trend_periods) + Momen
            data["Momentum" + str(trend_periods)] = Momen
            return data

        def gen_Larry(data, periods_long=14, open_col='Open', high_col='High', low_col='Low', close_col='Close'):
            Close = data[close_col]
            Open = data[open_col]
            High = data[high_col]
            Low = data[low_col]
            Larry = []
            length = len(Close)
            for i in range(periods_long, length):
                Hn = High[i:i + periods_long].max()
                Ln = Low[i:i + periods_long].min()
                L = (Hn - Close[i]) * 100 / (Hn - Ln)
                Larry.append(L)
            Larry = [np.nan] * periods_long + Larry
            data["Larry" + str(periods_long)] = Larry
            return data

        def gen_KandD_percent(df, n=9):
            low_list = df['Low'].rolling(9, min_periods=9).min()
            low_list.fillna(value=df['Low'].expanding().min(), inplace=True)
            high_list = df['High'].rolling(9, min_periods=9).max()
            high_list.fillna(value=df['High'].expanding().max(), inplace=True)
            rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
            df['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
            df['D'] = df['K'].ewm(com=2).mean()
            df['J'] = 3 * df['K'] - 2 * df['D']
            return df

        def gen_CCI(data, n=5, high_col='High', low_col='Low', close_col='Close'):
            M = np.array((data[high_col] + data[low_col] + data[close_col]) / 3)
            N = np.ones(n)
            w = N / n
            SM = np.convolve(w, M)[n - 1:-n + 1]
            SM = np.array(SM)
            SM = np.r_[np.zeros(n - 1), SM]
            D = []
            for i in range(0, n):
                D.append(0)
            for i in range(n, M.shape[0]):
                d = 0
                for j in range(0, n):
                    e = abs(M[i - j] - SM[i])
                    d = d + e
                d = d / n
                D.append(d)
            D = np.array(D)
            D = D + 0.001
            CCI = (M - SM) / D / 0.015
            data["CCI" + str(n)] = CCI
            return data
        df = gen_simple_ma(df=df)
        df = gen_ewma(df=df)
        df = gen_RSI(df=df)
        df = gen_ADO(data=df)
        df = gen_macd(data=df)
        df = gen_average_true_range(data=df)
        df = gen_Momentum(data=df)
        df = gen_Larry(data=df)
        df = gen_KandD_percent(df=df)
        df = gen_CCI(data=df)

    #get featuresnames
    featuresnames = df.columns.values.tolist()

    # generate label
    def gen_label_step(data, price='Close', k=1, d=0.01, n=5):
        p0 = np.array(data[price])
        p1 = np.delete(np.append(p0,np.zeros(5)), [0,1,2,3,4])
        l = p1 / p0
        label = np.zeros(l.shape[0])
        for i in range(0, n):
            label0 = (l > (k + i * d)) + 0
            label = label + label0
        data["label"] = label
        return data

    def gen_label_point(data, point, price='Close'):
        p0 = np.array(data[price])
        p1 = np.delete(np.append(p0, 0), 0)
        l = p1 / p0
        label = np.zeros(l.shape[0])
        for i in range(0,len(point)):
            label0 = (l>point[i]) + 0
            label = label + label0
        data["label"] = label
        return data

    if steplabel == True:
        df = gen_label_step(data=df,k=k,d=d,n=n)
    else:
        df = gen_label_point(data=df,point=point)

    classnames = []
    if steplabel==True:
        n = len(point)
    for i in range(0, n + 1):
        classnames.append(str(i))

    #drop N/A
    df.dropna(inplace=True)

    X = np.array(df[featuresnames])
    Y = np.array(df["label"]).reshape(-1, 1)

    # Seperate train set
    X_train = X[t0:t1, :]
    X_test = X[t1:X.shape[0], :]
    Y_train = Y[t0:t1, :]
    Y_test = Y[t1:Y.shape[0], :]

    # Random forest
    RF0 = RF(max_depth=15, min_samples_leaf=15)
    RF0.fit(X_train, Y_train)
    accuracy0 = RF0.score(X_test, Y_test)
    print(accuracy0)
    Estimators = RF0.estimators_
    if output == True:
        for index, model in enumerate(Estimators):
            filenames = 'RF' + str(index)
            dot_data = skl.tree.export_graphviz(model, out_file=None,
                                                filled=True, rounded=True,
                                                special_characters=True,
                                                feature_names=featuresnames,
                                                class_names=classnames)
            graph = graphviz.Source(dot_data)
            graph.render(filename=filenames)

filename = ("C:/Users/lYXYl/Desktop/EECS545/project/test2.csv")
point = (0.96,1,1.04)
RF_regression(filename=filename,t0=0,t1=4700,point=point,k=0.96,d=0.02,n=5,steplabel=False,generateindices=True,output=True)