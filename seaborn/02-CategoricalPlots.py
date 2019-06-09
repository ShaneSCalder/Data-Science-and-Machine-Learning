#!/usr/bin/env python
# coding: utf-8

# In[12]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
tips = sns.load_dataset('tips')
tips.head()


# In[13]:


import numpy as np


# In[14]:


sns.barplot(x = 'sex', y = 'total_bill', data = tips, estimator=np.std)


# In[15]:


sns.countplot(x='sex',data=tips)


# In[16]:


sns.boxplot(x='day',y='total_bill',data=tips,hue='smoker')


# In[17]:


sns.boxplot(x='day',y='total_bill',data=tips,)


# In[19]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex')


# In[20]:


sns.violinplot(x='day',y='total_bill',data=tips)


# In[22]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex',split=True)


# In[25]:


sns.stripplot(x='day',y='total_bill',data=tips)


# In[26]:


sns.stripplot(x='day',y='total_bill',data=tips,hue='sex')


# In[28]:


#use doge instead of split
sns.stripplot(x='day',y='total_bill',data=tips,hue='sex',dodge=True)


# In[29]:


#do not use for large data sets
sns.swarmplot(x='day',y='total_bill',data=tips)


# In[30]:


sns.violinplot(x='day',y='total_bill',data=tips)
sns.swarmplot(x='day',y='total_bill',data=tips,color='black')


# In[34]:


sns.catplot(x='day',y='total_bill',data=tips,kind='boxen')


# In[40]:


sns.boxenplot(x='day',y='total_bill',data=tips,hue='sex',dodge=True)
sns.stripplot(x='day',y='total_bill',data=tips,hue='sex',dodge=True,color='black')


# In[ ]:




