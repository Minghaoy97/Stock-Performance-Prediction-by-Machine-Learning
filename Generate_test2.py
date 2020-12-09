#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


df = pd.read_csv('AAPL_combined.csv')
df.dropna(axis=0, how='any', inplace=True)

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


# In[ ]:




