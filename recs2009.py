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


# one-hot-encode the categorical attributes
recs_enc_df = pd.get_dummies(recs_df,columns=features)

# create the features dataframe
recs_enc_features_df = recs_enc_df.drop('KWH',axis=1,inplace=False)

# create a list of the feature names from the column headers
recs_enc_features = list(recs_enc_features_df.columns.values)

# define X to be the features dataframe
X = recs_enc_features_df

# define Y to be the target
#Y = recs_enc_df[target]
#### group the KWH energy usage into bins

conditions = [
     (recs_enc_df['KWH'] >= 0)     & (recs_enc_df['KWH'] < 6000),
     (recs_enc_df['KWH'] >= 6000)  & (recs_enc_df['KWH'] < 9500),
     (recs_enc_df['KWH'] >= 9500)  & (recs_enc_df['KWH'] < 15000),
     (recs_enc_df['KWH'] >= 15000)]
choices=[0,1,2,3]

recs_enc_df['KWH_bin'] = np.select(conditions, choices, default=3)
Y = recs_enc_df[['KWH_bin']]


# create the train and test datasets with a split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=6202)



print("Gini importance")
## Feature Importance
### Based on max_depth plot, depth = 6 is most ideal
treereg = DecisionTreeRegressor(max_depth=6, random_state=6103)
treereg.fit(X_train, y_train)


### "Gini importance" of each feature: 
gini_features_df=(pd.DataFrame({'feature':recs_enc_features, 'importance':sorted(treereg.feature_importances_ *1000, reverse = True)}))


## plot the gini index data
fig = plt.figure(figsize=(15,10))
ax = plt.gca()
gini_top30 = gini_features_df.nlargest(30,'importance')
fig.add_subplot(1,1,1)
plt.bar(gini_top30['feature'], gini_top30['importance'])
ax.set_xticks(np.arange(len(gini_top30['feature'])))
ax.set_xticklabels(gini_top30['feature'],rotation = 75, ha="right")
ax.set_ylabel('Importance')
ax.set_title('Gini feature importance')
plt.show()
    

from heapq import nlargest
from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from sklearn.neighbors import KNeighborsClassifier


def plot_features(feature_list,reg_type):
    df = pd.DataFrame(feature_list)
    fig = plt.figure(figsize=(15,10))
    ax = plt.gca()

    fig.add_subplot(1,1,1)
    plt.bar(ind, df[0])
    ax.set_xticks(np.arange(len(df[1])))
    ax.set_xticklabels(df[1],rotation = 75, ha="right")
    ax.set_ylabel('Importance')
    ax.set_title(reg_type + ' feature importance')
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


# grab unique listing of attributes
def get_attribute_listing(data):
   temp_df = pd.DataFrame(data)[1].str.extractall(r'(([\w]+)_[^_]+$)')
   return pd.DataFrame(temp_df[1].drop_duplicates())


def one_hot_encode(data):
    s1=list(data.columns.values)
    return pd.get_dummies(data,columns=list(set(s1).intersection(set(features))))


def kn_classifier(data,type):
    temp_energy_df = recs_df[get_attribute_listing(data)[1]].copy()
    temp_energy_enc_df = one_hot_encode(temp_energy_df)
    
    # create the train and test datasets with a split
    tmpX_train, tmpX_test, tmpy_train, tmpy_test = train_test_split( temp_energy_enc_df, Y, test_size=0.3, random_state=6202)
    
    
    
    ## KNN-Tuning -->
    ### Determines what number should K should be
    
    ### Store results
    train_accuracy = []
    test_accuracy  = []
    ### Set KNN setting from 1 to 15
    knn_range = range(1, 15)
    for neighbors in knn_range:
    ### Start Nearest Neighbors Classifier with K of 1
      knn = KNeighborsClassifier(n_neighbors=neighbors,
                                  metric='minkowski', p=1)
      ### Train the data using Nearest Neighbors
      knn.fit(tmpX_train, tmpy_train)
      ### Capture training accuracy
      train_accuracy.append(knn.score(tmpX_train, tmpy_train))
      ### Predict using the test dataset  
      #Y_pred = knn.predict(tmpX_test)
      ### Capture test accuracy
      test_accuracy.append(knn.score(tmpX_test, tmpy_test))
      
    ## Plot Results from KNN Tuning
    plt.plot(knn_range, train_accuracy, label='training accuracy')
    plt.plot(knn_range, test_accuracy,  label='test accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Neighbors')
    plt.legend()
    plt.title('KNN Tuning ( '+type+' )')
    #plt.savefig('KNNTuning.png')
    plt.show()


    
    
# RANDOM FOREST Regressor
print("RandomForestRegressor")
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=10, max_features=4,n_jobs=2)
rf.fit(X_train, y_train)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), recs_enc_features), 
             reverse=True))
rand_forest = nlargest(30, sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), recs_enc_features), 
             reverse=True))
plot_features(rand_forest,"Random Forest")
kn_classifier(rand_forest,"Random Forest")



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

rfe_data = nlargest(30, sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), recs_enc_features), reverse=True))
plot_features(rfe_data,"RFE")
kn_classifier(rfe_data,"RFE")


ind = np.arange(0, 30)



print("") 
print("Linear Regression")
lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
lr_data = rank_to_dict(np.abs(lr.coef_), recs_enc_features, True)
print(lr_data)

plot_features(lr_data,"Linear Regression")
kn_classifier(lr_data,"Linear Regression")


print("") 
print("Ridge Regression")
ridge = Ridge(alpha=7)
ridge.fit(X_train, y_train)
ridge_data = (rank_to_dict(np.abs(ridge.coef_), recs_enc_features, True))
print(ridge_data)

plot_features(ridge_data,"Ridge Regression")
kn_classifier(ridge_data,"Ridge Regression")


print("")
print("Lasso")
lasso = Lasso(alpha=.00075,max_iter=1000)
lasso.fit(X_train, y_train)
lasso_data = (rank_to_dict(np.abs(lasso.coef_), recs_enc_features, False))
print(lasso_data)

plot_features(lasso_data,"Lasso")
kn_classifier(lasso_data,"Lasso")


print("")
print("RandomForestRegressor")
rf = RandomForestRegressor(n_jobs=2)
rf.fit(X,Y)
random_forest_data = (rank_to_dict(rf.feature_importances_, recs_enc_features, False))
print(random_forest_data)

plot_features(random_forest_data,"Random Forest Regressor")
kn_classifier(random_forest_data,"Random Forest Regressor")




