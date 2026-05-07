# -*- coding: utf-8 -*-
"""
@author: greg2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_curve, roc_auc_score)

y = pd.read_csv('y_preprocessed.csv').squeeze()
X = np.load('Xr_preprocessed.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr = LogisticRegression()

param_grid = {"C" : np.logspace(-4, 4, 20),
              "solver" : ['lbfgs', 'liblinear', 'sag']}

lr_cv = GridSearchCV(lr, param_grid, cv=5)
lr_cv.fit(X_train, y_train)

print(lr_cv.best_params_)
print(lr_cv.best_score_)

best_lr = LogisticRegression(max_iter=100,
                             C=lr_cv.best_params_['C'],
                             solver=lr_cv.best_params_['solver'])

#pour meta-modeles
from pickle import dump
with open("lr_dep_formetamod.pkl", "wb") as f:
    dump(best_lr, f, protocol=5)

best_lr.fit(X_train, y_train)
y_pred = best_lr.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

y_pred_prob = best_lr.predict_proba(X_test)
fpr1, tpr1, thresholds1 = roc_curve(y_test==1, y_pred_prob[:,1])
fpr0, tpr0, thresholds0 = roc_curve(y_test==0, y_pred_prob[:,0])
plt.plot([0,1], [0,1], "--")
plt.plot(fpr1, tpr1, label="Non Depression")
plt.plot(fpr0, tpr0, label="Depression")
plt.legend()
plt.xlabel("Taux faux positifs")
plt.ylabel("Taux vrai positifs")
plt.title("courbe ROC depression LogReg")
plt.show()

score_auc1 = roc_auc_score(y_test, y_pred_prob[:,1])
score_auc0 = roc_auc_score(y_test, y_pred_prob[:,0])
print(f"score AUC pour la classe 1 (Non Depression): {score_auc1}")
print(f"score AUC pour la classe 0 (Depression): {score_auc0}")