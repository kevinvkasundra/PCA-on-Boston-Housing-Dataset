# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:49:45 2019

@author: Kevin
"""

#PCA on the Boston Data

import pandas as pd

path = ('') #cause not all pcs have same path :P
df = pd.read_csv(path + 'train.csv') #importing the train file as Dataframe

df.describe() #summarize the data first
df.isnull().sum() #Find the NAN values, here fortunately zero

df = df.rename(columns ={'medv': 'price'}) #renaming medv with price (target)

y = df['price'].values
df.columns
x = df.drop(['ID','price'], axis=1).values

# Standardizing
#Its an important step to standardize the features before doing PCA
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)

# PCA projection
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data= principalComponents, columns = ['PC1', 'PC2','PC3', 'PC4', 'PC5'])
pca.explained_variance_ratio_
