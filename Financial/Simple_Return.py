#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Calculating Security's rate of simple return
import numpy as np
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import pandas_datareader.data as wb
import yfinance as yf


# In[3]:


#import the data
dataset = wb.get_data_yahoo('GOOG', start='2019-01-01', end='2019-05-31')


# In[4]:


dataset.head()


# In[5]:


dataset.tail()


# In[8]:


dataset['simple_return'] = (dataset['Adj Close'] / dataset['Adj Close'].shift(1)) - 1
print(dataset['simple_return'])


# In[10]:


#Avergae daily simple return
dataset['simple_return'].mean()


# In[11]:


dataset['simple_return'].std()


# In[ ]:




