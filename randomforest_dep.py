# -*- coding: utf-8 -*-
"""
@author: greg2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_curve, roc_auc_score)

y = pd.read_csv('y_preprocessed.csv').squeeze()
X = np.load('Xr_preprocessed.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier()

param_grid = {'criterion': ['gini', 'entropy'], 
              'max_depth': [None, 5, 10, 20, 50],
              'min_samples_leaf': [1, 10, 20],
              'min_samples_split': [2, 10, 20, 50]
              }

rf_cv = GridSearchCV(rf, param_grid, cv=3)
rf_cv.fit(X_train, y_train)

print(rf_cv.best_params_)
print(rf_cv.best_score_)

best_rf = RandomForestClassifier(criterion=rf_cv.best_params_['criterion'],
                                 max_depth=rf_cv.best_params_['max_depth'],
                                 min_samples_leaf=rf_cv.best_params_['min_samples_leaf'],
                                 min_samples_split=rf_cv.best_params_['min_samples_split'],
                                 )

#pour meta-modeles
from pickle import dump
with open("rf_dep_formetamod.pkl", "wb") as f:
    dump(best_rf, f, protocol=5)


best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

y_pred_prob = best_rf.predict_proba(X_test)
fpr1, tpr1, thresholds1 = roc_curve(y_test==1, y_pred_prob[:,1])
fpr0, tpr0, thresholds0 = roc_curve(y_test==0, y_pred_prob[:,0])
plt.plot([0,1], [0,1], "--")
plt.plot(fpr1, tpr1, label="Non Depression")
plt.plot(fpr0, tpr0, label="Depression")
plt.legend()
plt.xlabel("Taux faux positifs")
plt.ylabel("Taux vrai positifs")
plt.title("courbe ROC depression Random Forest")
plt.show()

score_auc1 = roc_auc_score(y_test, y_pred_prob[:,1])
score_auc0 = roc_auc_score(y_test, y_pred_prob[:,0])
print(f"score AUC pour la classe 1 (Non Depression): {score_auc1}")
print(f"score AUC pour la classe 0 (Depression): {score_auc0}")

