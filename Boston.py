# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:04:55 2019

@author: Kevin
"""

import pandas as pd
import sklearn 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

##### Data Import #####
boston = load_boston() #Loading the Boston Data File
print(boston.keys()) # Available Dictionary Keys
print(boston.data.shape) #Dimension of Data file
print(boston.feature_names)
print(boston.DESCR)

df = pd.DataFrame(boston.data) #Converting to Pandas
print(df.head()) #Observing data
df.columns = boston.feature_names
print(df.head()) #Observing data

print(boston.target.shape) #Check dimensions of target
df['price'] = boston.target #Add Target Price to the Dataframe
print(df.head())

print(df.describe()) #Summarizing the dataset

df.isnull().sum() #NAN values in each column

##### Exploration #####
from pandas.plotting import scatter_matrix
import seaborn as sns

plt.plot(df) # variation of each attribute
scatter_matrix(df, figsize=(16,12), alpha=0.3) #ScatterPlot of attributes

print(df.corr()) #Correlation Matrix
sns.heatmap(df.corr()) #Heatmap showing Correlation as a function of colour



X = df.drop('price', axis=1)
Y = df['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, 
                                                    random_state =12345)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)
plt.scatter = (Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show
plt.figure()

