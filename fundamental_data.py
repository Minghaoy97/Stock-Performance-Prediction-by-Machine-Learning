#!/usr/bin/env python
# coding: utf-8

# In[41]:


import xlrd  

apple = xlrd.open_workbook('apple.xlsx')


# In[42]:


datasheet = apple.sheets()[0]
select_data = []
for i in range(datasheet.nrows):  
    if '' not in datasheet.row_values(i):
        select_data.append(datasheet.row_values(i))
        


# In[43]:


for i in range(len(select_data)):
    for j in range(len(select_data[i])):
        if select_data[i][j] == '-':
            select_data[i][j] = select_data[i][j - 1]


# In[44]:


import pandas as pd
from collections import defaultdict
from pandas.core.frame import DataFrame

fund_data = defaultdict(list)
for i in select_data:
    fund_data[i[0]] = i[1:]
data = DataFrame(fund_data)
data


# In[58]:





# In[ ]:




