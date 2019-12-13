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
from imblearn.over_sampling import RandomOverSampler

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
# # matplotlib and seaborn for plotting
# import matplotlib.pyplot as plt
# import seaborn as sns


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

# Correlation
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def normalise(vec):
    return (vec - vec.mean())/vec.std()

def data_preparation(df):
    # Date treatment
    # df['date_recorded'] = df['date_recorded'].apply(lambda x: pd.to_datetime(x)).astype(int)
    # df['date_recorded'].apply(lambda x: pd.to_datetime(x).day)

    # Nettoyer les données
    df = df.drop("id", axis=1)
    df = df.drop("extraction_type_group", axis=1) # extraction_type
    df = df.drop("quality_group", axis=1) # water_quality
    df = df.drop("quantity", axis=1) # "quantity_group" identitique
    df = df.drop("source_type", axis=1) # on garde source
    df["source"] = df["source"].replace("unknown", 'other')
    df = df.drop("waterpoint_type_group", axis=1) # On garde "waterpoint_type"
    df = df.drop("date_recorded", axis=1)

    # Numeric & non numeric columns
    numerics = ['int16', 'int32', 'int64',
                'float16', 'float32', 'float64']
    non_numeric_columns = df.select_dtypes(exclude=numerics).columns
    numeric_columns = df.select_dtypes(include=numerics).columns

    # df_all.isnull().sum()
    for col in numeric_columns:
        df[col].fillna(value=df[col].mean(), inplace=True)

    # Pre-traitement donnees
    # Construction_year
    no_construction_year = df["construction_year"]==0
    df["no_construction_year"] = no_construction_year*1

    # Difficile de choisir par quoi remplacer la valeur nulle
    # Choix Moyenne
    construction_year = df[df["construction_year"]!=0]
    df["construction_year"] = df["construction_year"].replace(0, construction_year["construction_year"].mean())

    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range = (0, 1))

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

    # Features engineering
    v = normalise(df["construction_year"])

    # Scaling data
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df),
                    columns=df.columns)
    df["nb_years"] = v

    return df


train = pd.read_csv('train.csv', header=0, sep=",")
test = pd.read_csv('test.csv', header=0, sep=",")

target = train["status_group"]
piv_train = train.shape[0]

# Resample
# Over-sampling
ros = RandomOverSampler(random_state=0)
train_resampled, target = ros.fit_resample(train.drop('status_group', axis=1), target)
piv_train = train_resampled.shape[0]

df_all = pd.concat((train_resampled, test), axis=0, ignore_index=True)

features_mask = ["amount_tsh", "funder", "gps_height", 'installer', 'num_private',
                 "basin", 'wpt_name', 'construction_year',
                 "permit", "extraction_type", "payment", "quantity", "source_class"]
df = df_all

## Model 1: 3 classes
df = data_preparation(df)

# Cross-Validation
X = df.iloc[0:piv_train].values
y = target.values

# A parameter grid for XGBoost
params = {
        'n_estimators':[100, 500],
        'min_child_weight': [1, 3, 5],
        'gamma': [0.2, 0.5, 1, 2],
        'subsample': [0.4, 0.6, 0.8],
        'colsample_bytree': [0.4, 0.6, 0.8],
        'max_depth': [10, 20, 50]
       }

# Randomized search for XGBClassifier
num_class = len(target.unique())
xgb = XGBClassifier(learning_rate=0.02, objective='multi:softmax', num_class=num_class, silent=True, nthread=1)

folds = 3
param_comb = 20

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 2019)

random_search = RandomizedSearchCV(xgb,
                                  param_distributions=params,
                                  n_iter=param_comb,
                                  n_jobs=3,
                                  cv=skf.split(X, y),
                                  verbose=3,
                                  random_state=2019)

# start = time()
random_search.fit(X, y)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

cv_accuracy = 0.

for train_index, cv_index in skf.split(X, y):
    X_train, X_cv = X[train_index], X[cv_index]
    y_train, y_cv = y[train_index], y[cv_index]
    lr_1 = XGBClassifier(learning_rate=0.02,
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
    lr_1.fit(X_train, y_train)
    y_pred = lr_1.predict(X_cv)
    cv_accuracy += accuracy_score(y_cv, y_pred)

print('CV accuracy: ' + str(cv_accuracy / 3.))

# Build prediction
lr_1 = XGBClassifier(learning_rate=0.02,
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

lr_1 = XGBClassifier(learning_rate=0.02,
                    n_estimators=500,
                    objective='multi:softmax',
                    num_class=3,
                    silent=True,
                    nthread=1,
                    min_child_weight=1,
                    gamma=0.5,
                    subsample=0.4,
                    colsample_bytree=0.6,
                    max_depth=20)

# Fitting
lr_1.fit(df.iloc[0:piv_train], target) # train

# Prédictions
last = df.shape[0]
test["status_group"] = lr_1.predict(df.iloc[piv_train:last]) # test

# Ecriture fichier
t = test.replace("functional", 0).replace("non functional", 1).replace("functional needs repair", 2)
t[['id', 'status_group']].to_csv('minimal_submission.csv', index=False)

