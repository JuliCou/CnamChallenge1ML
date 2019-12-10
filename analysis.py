#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:48:53 2019

@author: courgibet
"""

import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from time import time
# import xgboost as xgb
from xgboost import XGBClassifier

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns



# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


train = pd.read_csv('train.csv', header=0, sep=",")
test = pd.read_csv('test.csv', header=0, sep=",")

target = train["status_group"]
piv_train = train.shape[0]
df_all = pd.concat((train.drop('status_group', axis=1), test), axis=0, ignore_index=True)

features_mask = ["amount_tsh", "funder", "gps_height", 'installer', 'num_private',
                 "basin", 'wpt_name', 'construction_year',
                 "permit", "extraction_type", "payment", "quantity", "source_class"]
df = df_all

# Date treatment
# df['date_recorded'] = df['date_recorded'].apply(lambda x: pd.to_datetime(x)).astype(int)
df['date_recorded'].apply(lambda x: pd.to_datetime(x).day)

# Numeric & non numeric columns
numerics = ['int16', 'int32', 'int64',
            'float16', 'float32', 'float64']
non_numeric_columns = df.select_dtypes(exclude=numerics).columns
numeric_columns = df.select_dtypes(include=numerics).columns

# Nettoyer les données
# df_all.isnull().sum()
for col in numeric_columns:
    df[col].fillna(value=df[col].mean(), inplace=True)

# Pre-traitement donnees
# Construction_year
no_construction_year = df["construction_year"]==0
df["no_construction_year"] = no_construction_year*1

# Difficile de choisir par quoi remplacer la valeur nulle
# Choix Moyenne
# construction_year = df[df["construction_year"]!=0]
# df["construction_year"].replace(0, construction_year["construction_year"].mean())


# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (-1, 1))

# Encoder valeurs non numériques
# df[non_numeric_columns] = df[non_numeric_columns].astype(str).apply(LabelEncoder().fit_transform)
le = LabelEncoder()
le_count = 0
dummies_columns = []
for col in non_numeric_columns:
    if len(df[col].unique()) > 20 or len(df[col].unique()) <=2 :
        le.fit(df[col].astype(str))
        df[col] = le.transform(df[col].astype(str))
        le_count += 1
    else:
        dummies_columns.append(col)

d = pd.get_dummies(df[dummies_columns])
df[d.columns]=d
df = df.drop(dummies_columns, axis=1)
df = df.drop("id", axis=1)

# Scaling data
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df),
                  columns=df.columns)

# Cross-Validation
X = df.iloc[0:piv_train].values
y = target.values


# param for RandomForestClassifier
# param_dist = {"max_depth": [3, None],
#               "max_features": sp_randint(1, df.shape[1]),
#               "min_samples_split": sp_randint(2, df.shape[1]),
#               "bootstrap": [False],
#               "criterion": ["gini", "entropy"]}

# A parameter grid for XGBoost
params = {
        'n_estimators':[50, 100, 200, 500],
        'min_child_weight': [1, 3, 5],
        'gamma': [0.2, 0.5, 1, 2],
        'subsample': [0.4, 0.6, 0.8],
        'colsample_bytree': [0.4, 0.6, 0.8],
        'max_depth': [1, 5, 10, 20]
       }

# run randomized search for RandomForestClassifier
# n_iter_search = 20
# lr = RandomForestClassifier(n_estimators=20)
# random_search = RandomizedSearchCV(lr, param_distributions=param_dist,
#                                    n_iter=n_iter_search, cv=5, iid=False)

# Randomized search for XGBClassifier
num_class = len(target.unique())
xgb = XGBClassifier(learning_rate=0.02, objective='multi:softmax',
                    num_class=num_class, silent=True, nthread=1)

folds = 3
param_comb = 30

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 2019)

random_search = RandomizedSearchCV(xgb,
                                  param_distributions=params,
                                  n_iter=param_comb,
                                  n_jobs=4,
                                  cv=skf.split(X, y),
                                  verbose=3,
                                  random_state=2019)

# start = time()
random_search.fit(X, y)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# kf = StratifiedKFold(n_splits=3, random_state=2019)
cv_accuracy = 0.

for train_index, cv_index in skf.split(X, y):
    X_train, X_cv = X[train_index], X[cv_index]
    y_train, y_cv = y[train_index], y[cv_index]
    # lr = LogisticRegression(solver='liblinear', multi_class='auto')
    # lr = RandomForestClassifier(n_estimators=20,
    #                             max_depth=random_search.best_estimator_.max_depth,
    #                             max_features=random_search.best_estimator_.max_features,
    #                             min_samples_split=random_search.best_estimator_.min_samples_split,
    #                             bootstrap=random_search.best_estimator_.bootstrap,
    #                             criterion=random_search.best_estimator_.criterion)
    lr = XGBClassifier(learning_rate=0.02,
                       n_estimators=random_search.best_estimator_.n_estimators,
                       objective='multi:softmax',
                       num_class=num_class,
                       silent=True,
                       nthread=1,
                       min_child_weight=random_search.best_estimator_.min_child_weight,
                       gamma=random_search.best_estimator_.gamma,
                       subsample=random_search.best_estimator_.subsample,
                       colsample_bytree=random_search.best_estimator_.colsample_bytree,
                       max_depth=random_search.best_estimator_.max_depth)
    # lr = XGBClassifier(learning_rate=0.02,
    #                    n_estimators=random_search.best_estimator_.n_estimators,,
    #                    objective='multi:softmax',
    #                    num_class=num_class,
    #                    silent=True,
    #                    nthread=1,
    #                    min_child_weight=1,
    #                    gamma=0.5,
    #                    subsample=0.8,
    #                    colsample_bytree=0.6,
    #                    max_depth=10)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_cv)
    cv_accuracy += accuracy_score(y_cv, y_pred)

print('CV accuracy: ' + str(cv_accuracy / 3.))


# Build prediction
# lr = LogisticRegression(solver='liblinear', multi_class='auto')
# lr = RandomForestClassifier() # score :  0.79696
# lr = XGBClassifier() # 0.74 (quelque chose)
# lr = RandomForestClassifier(n_estimators=20,
#                             max_depth=random_search.best_estimator_.max_depth,
#                             max_features=random_search.best_estimator_.max_features,
#                             min_samples_split=random_search.best_estimator_.min_samples_split,
#                             bootstrap=random_search.best_estimator_.bootstrap,
#                             criterion=random_search.best_estimator_.criterion)
# 0.81043

lr = XGBClassifier(learning_rate=0.02,
                    n_estimators=random_search.best_estimator_.n_estimators,
                    objective='multi:softmax',
                    num_class=num_class,
                    silent=True,
                    nthread=1,
                    min_child_weight=random_search.best_estimator_.min_child_weight,
                    gamma=random_search.best_estimator_.gamma,
                    subsample=random_search.best_estimator_.subsample,
                    colsample_bytree=random_search.best_estimator_.colsample_bytree,
                    max_depth=random_search.best_estimator_.max_depth)
lr.fit(df.iloc[0:piv_train], target) # train
last = df.shape[0]
test["status_group"] = lr.predict(df.iloc[piv_train:last]) # test
test[['id', 'status_group']].to_csv('minimal_submission_legend.csv', index=False)
test = test.replace("functional", 0).replace("non functional", 1).replace("functional needs repair", 2)
test[['id', 'status_group']].to_csv('minimal_submission.csv', index=False)


