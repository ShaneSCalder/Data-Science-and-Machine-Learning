#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Import Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv('kyphosis.csv')


# In[7]:


#Age is in months
df.head()


# In[8]:


df.info()


# In[9]:


sns.pairplot(df, hue='Kyphosis')


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X = df.drop('Kyphosis', axis=1)


# In[14]:


y = df['Kyphosis']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[16]:


from sklearn.tree import DecisionTreeClassifier


# In[17]:


dtree = DecisionTreeClassifier()


# In[19]:


dtree.fit(X_train,y_train)


# In[20]:


predictions = dtree.predict(X_test)


# In[21]:


from sklearn.metrics import classification_report,confusion_matrix


# In[24]:


print(confusion_matrix(y_test,predictions))
print("\n")
print(classification_report(y_test,predictions))


# In[36]:


#Tree visualization

from IPython.display import Image  
#from sklearn.externals.six import StringIO 
from six import StringIO
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[1:])
features


# In[37]:


dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png()) 


# In[26]:


from sklearn.ensemble import RandomForestClassifier


# In[31]:


rfc = RandomForestClassifier(n_estimators=200)


# In[32]:


rfc.fit(X_train,y_train)


# In[33]:


rfc_pred = rfc.predict(X_test)


# In[34]:


print(confusion_matrix(y_test,rfc_pred))
print("\n")
print(classification_report(y_test,rfc_pred))


# In[ ]:




