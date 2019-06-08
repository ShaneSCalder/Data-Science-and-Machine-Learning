#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


tips = sns.load_dataset('tips')


# In[5]:


tips.head()


# In[6]:


# Use kde for histrogram and bins for distribution of data
sns.distplot(tips['total_bill'],kde=False,bins=30)


# In[7]:


# joint plot needs 3 inputs x, y and data
# use kind to change from scatter kind=hex will turn to hex
sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')


# In[8]:


sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')


# In[9]:


sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde')


# In[10]:


sns.jointplot(x='total_bill', y='tip', data=tips)


# In[11]:


sns.pairplot(tips)


# In[12]:


sns.pairplot(tips, hue='sex')


# In[13]:


sns.pairplot(tips, hue='sex',palette='coolwarm')


# In[14]:


sns.rugplot(tips['total_bill'])


# In[15]:


sns.distplot(tips['total_bill'])


# In[16]:


sns.distplot(tips['total_bill'],kde=False)


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# In[18]:


#create data set 
dataset = np.random.randn(25)

#create rugplot
sns.rugplot(dataset);

#set up the x-axis for plot
x_min = dataset.min() - 2
x_max = dataset.max() + 2

# 100 equally spaced points from x_min to x_max
x_axis = np.linspace(x_min, x_max,100)

# Set up the bandwidth, for info on this:
url = 'http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth'

bandwidth = ((4*dataset.std()**5)/(3*len(dataset)))**.2


# Create an empty kernel list
kernel_list = []

# Plot each basis function
for data_point in dataset:
    
    # Create a kernel for each point and append to list
    kernel = stats.norm(data_point,bandwidth).pdf(x_axis)
    kernel_list.append(kernel)
    
    #Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * .4
    plt.plot(x_axis,kernel,color = 'grey',alpha=0.5)

plt.ylim(0,1)


# In[19]:


# To get the kde plot we can sum these basis functions.

# Plot the sum of the basis function
sum_of_kde = np.sum(kernel_list,axis=0)

# Plot figure
fig = plt.plot(x_axis,sum_of_kde,color='indianred')

# Add the initial rugplot
sns.rugplot(dataset,c = 'indianred')

# Get rid of y-tick marks
plt.yticks([])

# Set title
plt.suptitle("Sum of the Basis Functions")


# In[20]:


sns.kdeplot(tips['total_bill'])
sns.rugplot(tips['total_bill'])


# In[21]:


sns.kdeplot(tips['tip'])
sns.rugplot(tips['tip'])


# In[ ]:




