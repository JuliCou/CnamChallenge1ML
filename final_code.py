#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:48:53 2019

@author: courgibet
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


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

def data_preparation(df):
    # Date treatment
    df["year_recorded"] = df.date_recorded.apply(lambda x: pd.to_datetime(x).year)
    df["month_recorded"] = df.date_recorded.apply(lambda x: pd.to_datetime(x).month)
    df["season_recorded"] = (df["month_recorded"]%12 + 3)//3
    
    # Nettoyer les données redondantes
    df = df.drop("id", axis=1)
    df = df.drop("extraction_type_group", axis=1) # extraction_type
    df = df.drop("quality_group", axis=1) # water_quality
    df = df.drop("quantity", axis=1) # "quantity_group" identitique
    df = df.drop("source_type", axis=1) # on garde source
    df["source"] = df["source"].replace("unknown", 'other')
    df = df.drop("waterpoint_type_group", axis=1) # On garde "waterpoint_type"
    df = df.drop("date_recorded", axis=1)
    df = df.drop("recorded_by", axis=1)

    # Numeric & non numeric columns
    numerics = ['int16', 'int32', 'int64',
                'float16', 'float32', 'float64']
    non_numeric_columns = df.select_dtypes(exclude=numerics).columns
    numeric_columns = df.select_dtypes(include=numerics).columns

    for col in numeric_columns:
        df[col].fillna(value=df[col].mean(), inplace=True)

    for col in non_numeric_columns:
        df[col].fillna(value="undefined", inplace=True)
    
    # Pre-traitement donnees
    # Construction_year
    no_construction_year = df["construction_year"]==0
    df["no_construction_year"] = no_construction_year*1

    # Replace  by mean (median?)
    construction_year = df[df["construction_year"]!=0]
    df["construction_year"] = df["construction_year"].replace(0, construction_year["construction_year"].mean())
    # Feature engineering
    df['operation_time'] = df.year_recorded - df.construction_year
    
    # GPS height
    # On conserve valeurs dans les fichiers d'origine (proches de celles récupérées via gpsvisualizer)
    # On crée une colonne valeur nulle (peut renseigner sur un éventuel état)
    no_gps_height = df["gps_height"]==0
    df["no_gps_height"] = no_gps_height*1
    
    # Averaging longitude, latitude and gps_height for longitude == 0
    # On crée une colonne valeur nulle
    no_longitude = df["longitude"]==0
    df["no_longitude"] = no_longitude*1

    a = df[df["longitude"] == 0]
    a.iloc[:, df.columns == "latitude"] = np.nan
    a.iloc[:, df.columns == "longitude"] = np.nan
    a.iloc[:, df.columns == "gps_height"] = np.nan
    df[df["longitude"] == 0] = a
    df["longitude"] = df.groupby("region_code").transform(lambda x: x.fillna(x.mean())).longitude
    df["latitude"] = df.groupby("region_code").transform(lambda x: x.fillna(x.mean())).latitude
    df["gps_height"] = df.groupby("region_code").transform(lambda x: x.fillna(x.mean())).gps_height

    # Obtention du fichier de coordonnées avec gps height 
    # Recording values to be used
    # coordinates = df[["id", "latitude", "longitude", "gps_height"]]
    # coordinates.to_csv('coordinates.csv', index=False)
    # https://www.gpsvisualizer.com/elevation
    gps_height = pd.read_table("gps_height.txt")
    df["new_gps_height"] = gps_height["altitude (m)"]
    df["gps_height"] = df["gps_height"].replace(0, df["new_gps_height"])
    
    # Encoder valeurs non numériques
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

    # Scaling data each feature from 0-1
    scaler = MinMaxScaler(feature_range = (-2, 2))
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df),
                      columns=df.columns)
    
    return df

# Import data
train = pd.read_csv('train.csv', header=0, sep=",")
test = pd.read_csv('test.csv', header=0, sep=",")

target = train["status_group"]
piv_train = train.shape[0]

# Data prep'
df = pd.concat((train.drop('status_group', axis=1), test), axis=0, ignore_index=True)
df = data_preparation(df)

# Model
X = df.iloc[0:piv_train].values
y = target.values
# Finding best parameters
# A parameter grid for XGBoost
params = {
        'n_estimators':[300, 500, 700],
        'min_child_weight': [1, 3, 5],
        'gamma': [0.2, 0.5, 1, 2],
        'subsample': [0.4, 0.6, 0.8],
        'colsample_bytree': [0.4, 0.6, 0.8],
        'max_depth': [10, 20, 50]
       }

num_class = len(target.unique())
xgb = XGBClassifier(learning_rate=0.02, objective='multi:softmax',
                    num_class=num_class, silent=True, nthread=1)

folds = 3
param_comb = 15

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 2019)

random_search = RandomizedSearchCV(xgb,
                                   param_distributions=params,
                                   n_iter=param_comb,
                                   n_jobs=3,
                                   cv=skf.split(X, y),
                                   verbose=3,
                                   random_state=2019)

random_search.fit(X, y)
report(random_search.cv_results_)

# Cross-validation
cv_accuracy = 0.

for train_index, cv_index in skf.split(X, y):
    X_train, X_cv = X[train_index], X[cv_index]
    y_train, y_cv = y[train_index], y[cv_index]
    
    xgb = XGBClassifier(learning_rate=0.02,
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
                       
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_cv)
    cv_accuracy += accuracy_score(y_cv, y_pred)

print('CV accuracy: ' + str(cv_accuracy / 3.))

# Building final model
xgb = XGBClassifier(learning_rate=0.02,
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
xgb.fit(X, y)

# Predictions
last = df.shape[0]
test["status_group"] = xgb.predict(df.iloc[piv_train:last].values)

# Output files
test = test.replace("functional", 0).replace("non functional", 1).replace("functional needs repair", 2)
test[['id', 'status_group']].to_csv('final_submission.csv', index=False)
