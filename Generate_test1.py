#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

stock_data = input()
df = pd.read_csv(stock_data)
df.dropna(axis=0, how='any', inplace=True)


# In[79]:


df.dropna(axis=0, how='any', inplace=True)
funds = df.iloc[:, 19:]
col_mean = funds.mean(axis = 0)
# for i in range(len(col_mean)):
#     funds.iloc[:, i] = funds.iloc[:, i] - col_mean[i]
funds = funds.values


# In[80]:


import numpy as np
from scipy.linalg import sqrtm

covfunds = np.cov(funds, rowvar = 0) #calculate the covariance matrix
eigVals, eigVects = np.linalg.eig(np.mat(covfunds)) #compute eigenvalues and eigenvectors
L = np.diag(eigVals**(-0.5))


# In[78]:


format(np.dot(eigVects, np.diag(eigVals))[0].A.squeeze()[0], '.4f') == format(np.dot(covfunds, eigVects)[0].A.squeeze()[0], '.4f')


# In[76]:


format(np.dot(covfunds, eigVects)[0].A.squeeze()[0], '.4f')


# In[81]:


trans = []
funds = pd.DataFrame(funds)
for i in range(len(col_mean)):
    funds.iloc[:, i] = funds.iloc[:, i] - col_mean[i]
print(funds)
for i in range(len(funds)):
    y = np.dot(np.dot(L, eigVects.T), funds[i])
    trans.append(y.A.squeeze())
trans = np.array(trans)
trans


# In[82]:


base = df.iloc[:, :19]
base = base.values
res = []
for i in range(len(base)):
    res.append(np.r_[base[i], trans[i]])
res = pd.DataFrame(np.array(res))


# In[84]:


res.to_csv('test1.csv', index = False)


# In[2]:


# Use standarded funds and technical factors
funds = df.iloc[:, 19:]
techs = df.iloc[:, 6:19]
base = df.iloc[:, :6]
base = base.values
col_mean_f = funds.mean(axis = 0)
col_mean_t = techs.mean(axis = 0)
col_var_f = funds.var(axis = 0)
col_var_t = techs.var(axis = 0)
for i in range(len(col_mean_f)):
    funds.iloc[:, i] = (funds.iloc[:, i] - col_mean_f[i])/col_var_f[i]
for i in range(len(col_mean_t)):
    techs.iloc[:, i] = (techs.iloc[:, i] - col_mean_t[i])/col_var_t[i]
techs = techs.values
funds = funds.values
res = []
for i in range(len(base)):
    res.append(np.r_[base[i], techs[i], funds[i]])
res = pd.DataFrame(np.array(res))
res


# In[93]:


res.to_csv('test2.csv', index = False) # standardized data (volume ->)


# In[2]:


import numpy as np

funds = df.iloc[:, 19:]
col_mean = np.array(funds.mean(axis = 0))
for i in range(len(col_mean)):
    funds.iloc[:, i] = (funds.iloc[:, i] - col_mean[i])
funds = funds.values
covfunds = np.cov(funds, rowvar = 0) #calculate the covariance matrix
eigVals, eigVects = np.linalg.eig(np.mat(covfunds)) #compute eigenvalues and eigenvectors
eigValIndice = np.argsort(eigVals)            
n_eigValIndice = eigValIndice[-1:-(12+1):-1]
n_eigVect = eigVects[:,n_eigValIndice]
other_eigVect = eigVects[:,n_eigValIndice[-1] + 1:]
x_mean = np.dot(np.dot(col_mean.reshape(1, 87), eigVects.T[0].T).A.squeeze(), eigVects.T[0].T)
for i in range(1, 87):
    x_mean += np.dot(np.dot(col_mean.reshape(1, 87), eigVects.T[i].T).A.squeeze(), eigVects.T[i].T)

lowDDataMat=np.array(funds*n_eigVect).squeeze()
base = df.iloc[:, :19]
base = base.values
res = []
for i in range(len(base)):
    res.append(np.r_[base[i], lowDDataMat[i].real])
res = pd.DataFrame(np.array(res))


# In[11]:


res


# In[ ]:





# ### SVR

# In[9]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
import sklearn
from sklearn import neural_network
from sklearn.metrics import mean_squared_error


# In[ ]:


def gen_pair(df, shift_day=-1, train_num = 1000):
    x = df.iloc[:,7:]
    y = np.array(df.iloc[:, 4]).reshape(-1,1)
    y = pd.DataFrame(y)
    y = np.array(y.shift(shift_day)).reshape(-1,1)
    y = y[:shift_day]
    x = np.array(x).reshape(-1,x.shape[1])[:shift_day]
    x_train = x[:train_num]
    y_train = y[:train_num]
    x_test = x[train_num:]
    y_test = y[train_num:]
    return df,x,y,x_train,y_train,x_test,y_test

shift_day = -1
stock = "TSLA"
df,x,y,x_train,y_train,x_test,y_test = gen_pair(res, shift_day=shift_day,train_num=400)
model = svm.SVR(kernel="linear",C=1)
model.fit(x_train,y_train)
plt.figure(figsize=(15,7))
plt.plot(model.predict(x_test), "r",label="predicted")
plt.plot(y_test, label = "actual")
plt.legend()
plt.title(f"Predict {stock} for {-shift_day} days")


# In[2]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier as RF
import graphviz

def RF_regression(filename,featuresnames,classnames,t0,t1,k=1,d=0.01,n=5,output=False):
    df = pd.read_csv(filename)

    # Seperate Date
    df['Y'], df['M'], df['Da'] = df["Date"].str.split('/').str
    df['Y'].astype("int")
    df['M'].astype("int")
    df['Da'].astype("int")
    del df['Date']

    # generate label
    def gen_label(data, price='Close', k=1, d=0.01, n=5):
        p0 = np.array(data[price])
        p1 = np.delete(np.append(p0, 0), 0)
        l = p1 / p0
        label = np.zeros(l.shape[0])
        for i in range(0, n):
            label0 = (l > (k + i * d)) + 0
            label = label + label0
        data["label"] = label
        return data

    df = gen_label(data=df,k=k,d=d,n=n)

    #drop N/A
    df.dropna(inplace=True)

    X = np.array(df[featuresnames,'Y','M','Da'])
    Y = np.array(df["label"]).reshape(-1, 1)

    # Seperate train set
    X_train = X[t0:t1, :]
    X_test = X[t1:X.shape[0], :]
    Y_train = Y[t0:t1, :]
    Y_test = Y[t1:Y.shape[0], :]

    # Random forest
    RF0 = RF(max_depth=15, min_samples_leaf=20)
    RF0.fit(X_train, Y_train)
    accuracy0 = RF0.score(X_test, Y_test)
    print(accuracy0)
    Estimators = RF0.estimators_
    if output == True:
        for index, model in enumerate(Estimators):
            filename = 'RF' + str(index)
            dot_data = skl.tree.export_graphviz(model, out_file=None,
                                                filled=True, rounded=True,
                                                special_characters=True,
                                                feature_names=featurenames,
                                                class_names=classnames)
            graph = graphviz.Source(dot_data)
            graph.render(filename=filename)


# In[ ]:


RF_regression('test2.csv',featuresnames,classnames,t0,t1,output=False)

