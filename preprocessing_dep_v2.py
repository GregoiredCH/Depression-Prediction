import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (LabelEncoder, StandardScaler)

df = pd.read_csv("FullData.csv")

#On enlève la colonne d'id
df = df.drop("SEQN", axis=1)

#Check des données manquantes
df = df.replace('Missing', np.nan) #on sait que les données manquantes ont été flagged en Missing
print(df.select_dtypes(include=["str", "int", "float"]).isnull().sum().to_string())

#On décide de drop les colonnes où >50% des données sont manquantes
to_drop = ['pregnant', 'first_cancer_type', 'second_cancer_type', 'third_cancer_type',
           'arthritis_type', 'full_time_work', 'out_of_work', 'lifetime_alcohol_consumption']

for col in to_drop:
    df = df.drop(col, axis=1)

cat_col = df.select_dtypes(include=["str"]).columns.tolist()
num_col = df.select_dtypes(include=["int", "float"]).columns.tolist()

#Remplace les valeurs manquantes par plus fréquentes pour catégorielles
df[cat_col] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_col])
#Remplace par la médiane pour les numériques
df[num_col] = SimpleImputer(strategy='median').fit_transform(df[num_col])

#Type des données
print(df.dtypes.to_string())
encoder = LabelEncoder()

for col in cat_col:
    df[col] = encoder.fit_transform(df[col])

#la target est depression
X = df.drop("depression", axis=1)
y = df["depression"]

print(np.mean(X))
print(np.std(X))
plt.figure(figsize=(100, 20)) 
X.boxplot() #je déconseille de l'exécuter, c'est illisible et std et mean permettent déjà de voir
#à standardiser

#Réduction de dimensions
#Matrice de corrélation, seuil 0.02
corr_matrix = df.corr()
corr_target = corr_matrix['depression']
selected_features = corr_target[abs(corr_target) > 0.02].drop('depression')
X = X[selected_features.index]

#Importance des features, seuil 0.01
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'Variable': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(14, 30))
plt.barh(X.columns, rf.feature_importances_) 
plt.show()

important_features = importance_df[importance_df['Importance'] > 0.01]['Variable']
Xs = X[important_features]
print(Xs.dtypes)

#PCA
#On standardise d'abord les données comme identifié au-dessus
scaler = StandardScaler()
Xst = scaler.fit_transform(Xs)

from sklearn.decomposition import PCA
pca_full = PCA()
pca_full.fit(Xst)

plt.grid()
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('# of components')
plt.ylabel('Cumulative explained variance')
plt.show()

#Coude à 30
pca = PCA(n_components=30)
Xr = pca.fit_transform(Xst)
print(sum(pca.explained_variance_ratio_))

#fonctions pour récupérer le traitement
np.save('Xr_preprocessed.npy', Xr)
y.to_csv('y_preprocessed.csv', index=False)


# Sauvegarder le pipeline pour le formulaire
from pickle import dump
with open("scaler.pkl", "wb") as f:
    dump(scaler, f, protocol=5)
 
with open("pca.pkl", "wb") as f:
    dump(pca, f, protocol=5)






#Exploration de données
#boxplot Xs
Xs.boxplot(figsize=(16, 6), rot=90)
plt.show()

#graphique pour visualiser la fréquence de chaque classe
freq= df['depression'].value_counts()
freq.plot(kind='bar')


#Histogrammes de toutes les features
selected_features = Xs.columns.tolist()

df[selected_features].hist(figsize=(20, 25))
plt.tight_layout()
plt.show()


#Histogrammes de quelques features
df[['BMI', 'glucose', 'sleep_hours', 'sedentary_time']].hist(figsize=(15, 5))
plt.tight_layout()
plt.show()

df[['hemoglobin', 'HDL', 'pulse', 'platelet_count']].hist(figsize=(15, 5))
plt.tight_layout()
plt.show()

df[['prescriptions_count', 'drinks_past_year', 'drinks_per_occasion', 'current_cigarettes_per_day']].hist(figsize=(15, 5))
plt.tight_layout()
plt.show()


#pairplots
import seaborn as sns

sns.pairplot(data=df,
             x_vars=['BMI', 'glucose', 'prescriptions_count', 'drinks_past_year'],
             y_vars=['sleep_hours', 'sedentary_time', 'hemoglobin', 'HDL'],
             hue='depression',
             palette=['Red', 'Blue'],
             diag_kind=None)
plt.show()

#heatmap des features selectionnées
corr_matrix = df[selected_features + ['depression']].corr().round(2)
plt.figure(figsize=(16, 14))
sns.heatmap(data=corr_matrix, annot=True)
plt.show()






