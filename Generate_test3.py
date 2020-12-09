#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


df = pd.read_csv('AAPL_combined.csv')
df.dropna(axis=0, how='any', inplace=True)

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
res


# In[ ]:




