# -*- coding: utf-8 -*-
"""
@author: greg2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (classification_report, roc_curve, roc_auc_score)

y = pd.read_csv('y_preprocessed.csv').squeeze()
X = np.load('Xr_preprocessed.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Avec kNN
from pickle import load
with open("knn_dep_formetamod.pkl", "rb") as f:
      knn = load(f)
    
bc = BaggingClassifier(knn)

param_grid = {"n_estimators" : [55, 60, 70]}
bc_cv = GridSearchCV(bc, param_grid, cv=5)
bc_cv.fit(X_train, y_train)
print(bc_cv.best_params_)
print(bc_cv.best_score_)

best_bc = BaggingClassifier(estimator=knn, n_estimators=bc_cv.best_params_['n_estimators'])
best_bc.fit(X_train, y_train)
y_pred = best_bc.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

y_pred_prob = best_bc.predict_proba(X_test)
fpr1, tpr1, thresholds1 = roc_curve(y_test==1, y_pred_prob[:,1])
fpr0, tpr0, thresholds0 = roc_curve(y_test==0, y_pred_prob[:,0])
plt.plot([0,1], [0,1], "--")
plt.plot(fpr1, tpr1, label="Non Depression")
plt.plot(fpr0, tpr0, label="Depression")
plt.legend()
plt.xlabel("Taux faux positifs")
plt.ylabel("Taux vrai positifs")
plt.title("courbe ROC depression Bagging avec kNN")
plt.show()

score_auc1 = roc_auc_score(y_test, y_pred_prob[:,1])
score_auc0 = roc_auc_score(y_test, y_pred_prob[:,0])
print(f"score AUC pour la classe 1 (Non Depression): {score_auc1}")
print(f"score AUC pour la classe 0 (Depression): {score_auc0}")

#Avec Logistic Regression
from pickle import load
with open("lr_dep_formetamod.pkl", "rb") as f:
    lr = load(f)
    
bc = BaggingClassifier(lr)

param_grid = {"n_estimators" : [55, 60, 70]}
bc_cv = GridSearchCV(bc, param_grid, cv=5)
bc_cv.fit(X_train, y_train)
print(bc_cv.best_params_)
print(bc_cv.best_score_)

best_bc = BaggingClassifier(estimator=lr, n_estimators=bc_cv.best_params_['n_estimators'])
best_bc.fit(X_train, y_train)
y_pred = best_bc.predict(X_test)


report = classification_report(y_test, y_pred)
print(report)

y_pred_prob = best_bc.predict_proba(X_test)
fpr1, tpr1, thresholds1 = roc_curve(y_test==1, y_pred_prob[:,1])
fpr0, tpr0, thresholds0 = roc_curve(y_test==0, y_pred_prob[:,0])
plt.plot([0,1], [0,1], "--")
plt.plot(fpr1, tpr1, label="Non Depression")
plt.plot(fpr0, tpr0, label="Depression")
plt.legend()
plt.xlabel("Taux faux positifs")
plt.ylabel("Taux vrai positifs")
plt.title("courbe ROC depression Bagging avec LogisticRegression")
plt.show()

score_auc1 = roc_auc_score(y_test, y_pred_prob[:,1])
score_auc0 = roc_auc_score(y_test, y_pred_prob[:,0])
print(f"score AUC pour la classe 1 (Non Depression): {score_auc1}")
print(f"score AUC pour la classe 0 (Depression): {score_auc0}")

#Sauvegarder le modèle entraîné
from pickle import dump
with open("bc_dep.pkl", "wb") as f:
    dump(best_bc, f, protocol=5)