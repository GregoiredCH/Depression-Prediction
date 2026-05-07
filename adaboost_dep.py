# -*- coding: utf-8 -*-
"""
@author: Grégoire DE CORDOUE-HECQUARD et João Paulo FERNANDES GUERRA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (classification_report, roc_curve, roc_auc_score)

y = pd.read_csv('y_preprocessed.csv').squeeze()
X = np.load('Xr_preprocessed.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#AdaBoost estimateur Decision Tree par défaut
adb = AdaBoostClassifier()

param_grid = {'n_estimators': np.arange(1, 16)}

adb_cv = GridSearchCV(adb, param_grid, cv=5)
adb_cv.fit(X_train, y_train)

print(adb_cv.best_params_)
print(adb_cv.best_score_)

best_adb = AdaBoostClassifier(n_estimators=adb_cv.best_params_['n_estimators'])
best_adb.fit(X_train, y_train)
y_pred = best_adb.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

y_pred_prob = best_adb.predict_proba(X_test)
fpr1, tpr1, thresholds1 = roc_curve(y_test==1, y_pred_prob[:,1])
fpr0, tpr0, thresholds0 = roc_curve(y_test==0, y_pred_prob[:,0])
plt.plot([0,1], [0,1], "--")
plt.plot(fpr1, tpr1, label="Non Depression")
plt.plot(fpr0, tpr0, label="Depression")
plt.legend()
plt.xlabel("Taux faux positifs")
plt.ylabel("Taux vrai positifs")
plt.title("courbe ROC depression AdaBoost")
plt.show()

score_auc1 = roc_auc_score(y_test, y_pred_prob[:,1])
score_auc0 = roc_auc_score(y_test, y_pred_prob[:,0])
print(f"score AUC pour la classe 1 (Non Depression): {score_auc1}")
print(f"score AUC pour la classe 0 (Depression): {score_auc0}")

#AdaBoost avec LogisticRegression
from pickle import load
with open("lr_dep_formetamod.pkl", "rb") as f:
    lr = load(f)
    
adb = AdaBoostClassifier(lr)

param_grid = {'n_estimators': np.arange(1, 16)}

adb_cv = GridSearchCV(adb, param_grid, cv=5)
adb_cv.fit(X_train, y_train)

print(adb_cv.best_params_)
print(adb_cv.best_score_)

best_adb = AdaBoostClassifier(n_estimators=adb_cv.best_params_['n_estimators'])
best_adb.fit(X_train, y_train)
y_pred = best_adb.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

score_auc1 = roc_auc_score(y_test, y_pred_prob[:,1])
score_auc0 = roc_auc_score(y_test, y_pred_prob[:,0])
print(f"score AUC pour la classe 1 (Non Depression): {score_auc1}")
print(f"score AUC pour la classe 0 (Depression): {score_auc0}")
#suffit à voir que très mauvais : 0.5

#AdaBoost avec best Decision Tree
from pickle import load
with open("dt_dep_formetamod.pkl", "rb") as f:
    dt = load(f)
    
adb = AdaBoostClassifier(dt)

param_grid = {'n_estimators': np.arange(1, 16)}

adb_cv = GridSearchCV(adb, param_grid, cv=5)
adb_cv.fit(X_train, y_train)

print(adb_cv.best_params_)
print(adb_cv.best_score_)

best_adb = AdaBoostClassifier(dt, n_estimators=adb_cv.best_params_['n_estimators'])
best_adb.fit(X_train, y_train)
y_pred = best_adb.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

y_pred_prob = best_adb.predict_proba(X_test)
fpr1, tpr1, thresholds1 = roc_curve(y_test==1, y_pred_prob[:,1])
fpr0, tpr0, thresholds0 = roc_curve(y_test==0, y_pred_prob[:,0])
plt.plot([0,1], [0,1], "--")
plt.plot(fpr1, tpr1, label="Non Depression")
plt.plot(fpr0, tpr0, label="Depression")
plt.legend()
plt.xlabel("Taux faux positifs")
plt.ylabel("Taux vrai positifs")
plt.title("courbe ROC depression AdaBoost")
plt.show()

score_auc1 = roc_auc_score(y_test, y_pred_prob[:,1])
score_auc0 = roc_auc_score(y_test, y_pred_prob[:,0])
print(f"score AUC pour la classe 1 (Non Depression): {score_auc1}")
print(f"score AUC pour la classe 0 (Depression): {score_auc0}")
