#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Lending club Project 2007 - 2010 data 
#Evaluation of borrowers (prediction of not fully paid)
#Create a Decision Tree and a Random Forest Model 
#Import Libraries 
#Import Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


loans = pd.read_csv('loan_data.csv')


# In[9]:


loans.head()


# In[13]:


loans.info()


# In[6]:


sns.pairplot(loans, hue='credit.policy')


# In[14]:


loans.describe()


# In[24]:


plt.figure(figsize=(12,8))
loans[loans['credit.policy']==1]['fico'].hist(bins=40,
                                               color='blue', label='Credit Policy = 1', 
                                              alpha=0.6, edgecolor='black')
loans[loans['credit.policy']==0]['fico'].hist(bins=40,
                                               color='red', label='Credit Policy = 0', 
                                              alpha=0.6, edgecolor='black')
plt.legend()
plt.xlabel('Fico Score')


# In[26]:


plt.figure(figsize=(12,8))
loans[loans['not.fully.paid']==1]['fico'].hist(bins=40,
                                               color='blue', label='Not Fully Paid = 1', 
                                              alpha=0.6, edgecolor='black')
loans[loans['not.fully.paid']==0]['fico'].hist(bins=40,
                                               color='red', label='Not Fully Paid = 0', 
                                              alpha=0.6, edgecolor='black')
plt.legend()
plt.xlabel('Fico Score')


# In[37]:


plt.figure(figsize=(12,8))
sns.countplot(x='purpose', hue='not.fully.paid',data=loans,palette='icefire')


# In[44]:


sns.jointplot(x='fico',y='int.rate',data=loans,color='hotpink')


# In[47]:


plt.figure(figsize=(12,8))
sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy',
           col='not.fully.paid', palette='icefire')


# In[48]:


#Create a purpose column
cat_purpose = ['purpose']


# In[49]:


final_data = pd.get_dummies(loans,columns=cat_purpose,drop_first=True)


# In[50]:


#purpose colum created with 1 or 0 as values 
final_data.head()


# In[52]:


#train and test 
from sklearn.model_selection import train_test_split


# In[55]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[56]:


#Create Decision Tree
from sklearn.tree import DecisionTreeClassifier 


# In[57]:


dtree = DecisionTreeClassifier()


# In[58]:


dtree.fit(X_train,y_train)


# In[59]:


pred1 = dtree.predict(X_test)


# In[60]:


from sklearn.metrics import classification_report,confusion_matrix


# In[61]:


print(classification_report(y_test,pred1))


# In[62]:


print(confusion_matrix(y_test,pred1))


# In[71]:


#Training the random Forrest 
from sklearn.ensemble import RandomForestClassifier


# In[72]:


rfc = RandomForestClassifier(n_estimators=350)


# In[73]:


rfc.fit(X_train,y_train)


# In[74]:


# Predictions
pred1 = rfc.predict(X_test)


# In[75]:


print(classification_report(y_test,pred1))


# In[76]:


print(confusion_matrix(y_test,pred1))


# In[ ]:




