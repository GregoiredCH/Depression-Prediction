# -*- coding: utf-8 -*-
"""
@author: greg2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (classification_report, roc_curve, roc_auc_score)

y = pd.read_csv('y_preprocessed.csv').squeeze()
X = np.load('Xr_preprocessed.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()
param_grid = {"n_neighbors": np.arange(3, 15)}

knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train, y_train)

print(knn_cv.best_params_)
print(knn_cv.best_score_)

best_knn = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])

#pour meta-modeles
from pickle import dump
with open("knn_dep_formetamod.pkl", "wb") as f:
    dump(best_knn, f, protocol=5)

best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

y_pred_prob = best_knn.predict_proba(X_test)
fpr1, tpr1, thresholds1 = roc_curve(y_test==1, y_pred_prob[:,1])
fpr0, tpr0, thresholds0 = roc_curve(y_test==0, y_pred_prob[:,0])
plt.plot([0,1], [0,1], "--")
plt.plot(fpr1, tpr1, label="Non Depression")
plt.plot(fpr0, tpr0, label="Depression")
plt.legend()
plt.xlabel("Taux faux positifs")
plt.ylabel("Taux vrai positifs")
plt.title("courbe ROC depression KNN")
plt.show()

score_auc1 = roc_auc_score(y_test, y_pred_prob[:,1])
score_auc0 = roc_auc_score(y_test, y_pred_prob[:,0])
print(f"score AUC pour la classe 1 (Non Depression): {score_auc1}")
print(f"score AUC pour la classe 0 (Depression): {score_auc0}")

from pickle import dump
with open("knn_dep.pkl", "wb") as f:
    dump(best_knn, f, protocol=5)