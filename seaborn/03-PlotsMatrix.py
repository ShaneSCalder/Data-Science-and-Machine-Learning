#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
tips.head()


# In[2]:


flights.head()


# In[8]:


tc = tips.corr()
tc


# In[7]:


sns.heatmap(tc)


# In[9]:


sns.heatmap(tc,annot=True)


# In[10]:


sns.heatmap(tc,annot=True,cmap='coolwarm')


# In[15]:


fb = flights.pivot_table(index='month',columns='year',values='passengers')
fb


# In[19]:


sns.heatmap(fb,cmap='coolwarm',linecolor='white',linewidths=1)


# In[23]:


sns.clustermap(fb,cmap='coolwarm',standard_scale=1)


# In[ ]:




