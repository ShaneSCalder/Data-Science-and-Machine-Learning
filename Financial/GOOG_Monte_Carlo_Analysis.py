#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')
import yfinance as yf


# In[16]:


#import the data
dataset = wb.get_data_yahoo('GOOG', start='2015-01-01', end='2019-05-31')['Adj Close']


# In[17]:


dataset.head()


# In[18]:


dataset.tail()


# In[21]:


log_returns = np.log(1 + dataset.pct_change())


# In[22]:


log_returns.tail()


# In[23]:


dataset.plot(figsize=(14,8))


# In[24]:


log_returns.plot(figsize=(14,8))


# In[26]:


u = log_returns.mean()
u


# In[27]:


var = log_returns.var()
var


# In[28]:


stdev = log_returns.std()
stdev


# In[29]:


drift = u - (0.5 * var)
drift


# In[30]:


np.array(drift)


# In[31]:


np.array(stdev)


# In[32]:


#set up a 95% chance of occurance
norm.ppf(0.95)


# In[33]:


x = np.random.rand(10,2)
x


# In[34]:


norm.ppf(x)


# In[35]:


Z = norm.ppf(np.random.rand(10,2))
Z


# In[36]:


t_intervals = 1000
interations = 10


# In[39]:


daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, interations)))
daily_returns


# In[40]:


s_zero = dataset.iloc[-1]
s_zero


# In[41]:


price_list = np.zeros_like(daily_returns)


# In[42]:


price_list


# In[44]:


price_list[0] = s_zero
price_list


# In[45]:


for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1] * daily_returns[t]


# In[46]:


price_list


# In[48]:


plt.figure(figsize=(14,8))
plt.title('GOOG Google Stock Price Predictions')
plt.xlabel('Future Days Traded')
plt.ylabel('Stock Price USD')
plt.plot(price_list)


# In[ ]:




