import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

features_mask = ['amount_tsh', 'gps_height', 'population']


# Cross-Validation
X = train[features_mask].values
y = train['status_group'].values

kf = StratifiedKFold(n_splits=3, random_state=2019)
cv_accuracy = 0.
for train_index, cv_index in kf.split(X, y):
    X_train, X_cv = X[train_index], X[cv_index]
    y_train, y_cv = y[train_index], y[cv_index]

    lr = LogisticRegression(solver='liblinear', multi_class='auto')
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_cv)
    cv_accuracy += accuracy_score(y_cv, y_pred)

print('CV accuracy: ' + str(cv_accuracy / 3.))


# Build prediction
lr = LogisticRegression(solver='liblinear', multi_class='auto')
lr.fit(train[features_mask], train['status_group'])
test['status_group'] = lr.predict(test[features_mask])
test['status_group'] = test['status_group'].replace({'functional': 0, 'non functional': 1, 'functional needs repair': 2}) 
test[['id', 'status_group']].to_csv('minimal_submission.csv', index=False)