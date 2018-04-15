#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 23:47:49 2018

@author: jimgrund
"""


import os
os.chdir('/Users/jimgrund/Documents/GWU/machine learning/recs2009-final-project/data/') 

### Basic Packages
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder


target = ['KWH']
features = np.array(['EQUIPM','WALLTYPE','ROOFTYPE','YEARMADE','REGIONC','ADQINSUL','TYPEGLASS',
                     'WINDOWS','DOOR1SUM','FUELHEAT','AUDIT','AIA_Zone',
                     'CELLAR','CRAWL','CONCRETE','BASEFIN','BASEHEAT','BASECOOL',
                     'ATTIC','ATTICFIN','ATTCHEAT','ATTCCOOL','AIRCOND','COOLTYPE',
                     'PRKGPLC1','GARGHEAT','GARGCOOL','HIGHCEIL'
                     ])

bin_features = np.array(['TOTSQFT','TOTROOMS','BEDROOMS','MONEYPY','NHSLDMEM','NCOMBATH','NHAFBATH','ACROOMS','HEATROOM','HDD65','CDD65'])

raw_recs_df  = pd.read_csv('recs2009_public.csv', low_memory=False)

### Traditional Null Check
raw_recs_df.dropna(inplace=True)
### replace anything that's -2 with 100
raw_recs_df = raw_recs_df.replace(-2, 100)


recs_df = pd.DataFrame(raw_recs_df[np.concatenate((np.append(features,target),bin_features), axis=0)])
#recs_features_df = pd.DataFrame(raw_recs_df[features])
#recs_bin_features_df = pd.DataFrame(raw_recs_df[bin_features])
#recs_target_df = pd.DataFrame(raw_recs_df[target])




#from sklearn.feature_extraction import FeatureHasher
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#import copy
#from sklearn.metrics import log_loss
#r = RandomForestClassifier(n_estimators=25,max_depth=10)
#X_train_hash = copy.copy(X_train)
#X_test_hash = copy.copy(X_test)
#for i in range(X_train_hash.shape[1]):
#    X_train_hash.iloc[:,i]=X_train_hash.iloc[:,i].astype('str')
#for i in range(X_test_hash.shape[1]):
#    X_test_hash.iloc[:,i]=X_test_hash.iloc[:,i].astype('str')
#h = FeatureHasher(n_features=100,input_type="string")
#X_train_hash = h.transform(X_train_hash.values)
#X_test_hash = h.transform(X_test_hash.values)
#
#l = LogisticRegression()
#l.fit(X_train_hash,y_train)
#y_pred = l.predict_proba(X_test_hash)
#print(log_loss(y_test,y_pred))#0.4
#
#r.fit(X_train_hash,y_train)
#y_pred = r.predict_proba(X_test_hash)
#print(log_loss(y_test,y_pred))


recs_enc_df = pd.get_dummies(recs_df,columns=features)
recs_enc_features_df = recs_enc_df.drop('KWH',axis=1,inplace=False)
recs_enc_features = list(recs_enc_features_df.columns.values)

X = recs_enc_features_df
Y = recs_enc_df[target]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.9, random_state=42)

#enc = OneHotEncoder(categorical_features=bin_features,handle_unknown='error')
#enc.fit(X_train)
#recs_ohe_df = enc.transform(X_train)


## ensure all datatypes are object type
#try:
#    del recs_features_df
#except:
#    pass
#recs_features_df = pd.DataFrame()
#for key in features:
#    try:
#        recs_features_df[key] = pd.DataFrame(recs_df[features].astype(dtype= {key:"object"}))
#    except:
#        recs_features_df = pd.DataFrame(recs_df[features].astype(dtype= {key:"object"}))
#
#try:
#    del recs_bin_features_df    
#except:
#    pass
#recs_bin_features_df = pd.DataFrame()
#for key in bin_features:
#    try:
#        recs_bin_features_df[key] = pd.DataFrame(recs_df[bin_features].astype(dtype= {key:"int64"}))
#    except:
#        recs_bin_features_df = pd.DataFrame(recs_df[bin_features].astype(dtype= {key:"int64"}))
#
#del recs_df
#recs_df = pd.concat([recs_features_df, recs_bin_features_df], ignore_index='True', axis=1)
#recs_features = np.concatenate((features,bin_features))

#from keras.utils import to_categorical    
#recs_features_df = to_categorical(recs_features_df)
#recs_features_df = x_df = pd.get_dummies(recs_features_df)

# one-hot encode the factors in x_df
#recs_df = to_categorical(recs_df)

#
#X = recs_df
#Y = raw_recs_df[target]
#names = np.array(recs_features)
#
#X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.8, random_state=42)


print("Gini importance")
## Feature Importance
### Based on max_depth plot, depth = 6 is most ideal
treereg = DecisionTreeRegressor(max_depth=6, random_state=6103)
treereg.fit(X_train, y_train)


### "Gini importance" of each feature: 
print(pd.DataFrame({'feature':recs_enc_features, 'importance':sorted(treereg.feature_importances_ *1000, reverse = True)}))





# RANDOM FOREST Regressor
print("RandomForestRegressor")
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=10, max_features=4)
rf.fit(X_train, y_train)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), recs_enc_features), 
             reverse=True))




# Recursive Feature Elimination
#this section takes about half hour to finish
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 8 attributes
rfe = RFE(model, 8)
rfe = rfe.fit(X_train, y_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), recs_enc_features)))


ind = np.arange(0, 30)

from heapq import nlargest
from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)

def plot_features(feature_list,reg_type):
    df = pd.DataFrame(feature_list)
    fig = plt.figure(figsize=(15,15))
    ax = plt.gca()

    fig.add_subplot(1,1,1)
    plt.bar(ind, df[0])
    ax.set_xticks(np.arange(len(df[1])))
    ax.set_xticklabels(df[1],rotation = 75, ha="right")
    ax.set_ylabel('Importance')
    ax.set_title(type + ' feature importance')
    plt.show()

def rank_to_dict(ranks, names, scaling, order=1):
    minmax = MinMaxScaler()
    #std = StandardScaler()
    if (scaling):
        ranks = minmax.fit_transform(order*np.array(ranks).T).T[0]
    else:
        ranks = order*np.array(ranks).T
    ranks = map(lambda x: round(x, 2), ranks)
    #return dict(zip(names, ranks ))
    return nlargest(30,sorted(zip(ranks,names ),reverse=True))

print("") 
print("Linear Regression")
lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
lr_data = rank_to_dict(np.abs(lr.coef_), recs_enc_features, True)
print(lr_data)

plot_features(lr_data,"Linear Regression")

print("") 
print("Ridge Regression")
ridge = Ridge(alpha=7)
ridge.fit(X_train, y_train)
ridge_data = (rank_to_dict(np.abs(ridge.coef_), recs_enc_features, True))
print(ridge_data)

plot_features(ridge_data,"Ridge Regression")

print("")
print("Lasso")
lasso = Lasso(alpha=1,max_iter=1000)
lasso.fit(X_train, y_train)
lasso_data = (rank_to_dict(np.abs(lasso.coef_), recs_enc_features, False))
print(lasso_data)

plot_features(lasso_data,"Lasso")

print("")
print("RandomForestRegressor")
rf = RandomForestRegressor()
rf.fit(X,Y)
random_forest_data = (rank_to_dict(rf.feature_importances_, recs_enc_features, False))
print(random_forest_data)

plot_features(random_forest_data,"Random Forest Regressor")
