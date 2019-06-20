#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:22:56 2019

@author: shane
"""
# Simple Linear Regression model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pandas_datareader.data as wb
import yfinance as yf
#$ pip install yfinance --upgrade --no-cache-dir

#import the data
dataset = wb.get_data_yahoo('GOOG', start='2019-01-01', end='2019-05-31')
#dataset = pd.read_csv('Simple_linear_Regression/Salary_Data.csv')
X = dataset.iloc[:, 3].values
y = dataset.iloc[:, 2].values

# Split into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state =0)

#Fitting LR to training set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X.reshape(-1, 1),y)
regressor.fit(X_train.reshape(-1,1), y_train)


# Predicting Test data set results 
y_pred = regressor.predict(X_test.reshape(-1,1))

#Visualizing the training results 
plt.scatter(X_train, y_train, color ='red')
plt.plot(X_train, regressor.predict(X_train.reshape(-1,1)), color = 'blue')
plt.title('Open VS Low(training set)')
plt.xlabel('Open')
plt.ylabel('Low')
plt.show()

#Visualizing the test results 
plt.scatter(X_test, y_test, color ='red')
plt.plot(X_test, regressor.predict(X_test.reshape(-1,1)), color = 'blue')
plt.title('Open VS Low(test set)')
plt.xlabel('Open')
plt.ylabel('Low')
plt.show()
