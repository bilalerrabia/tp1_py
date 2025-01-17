import seaborn as sns
iris=sns.load_dataset('iris')

print(iris.head())
print('_'*50)
print(iris.info())
print('_'*50)
print(iris.describe())
print('_'*50)

print(f"le nombre d'échantillonsech :{iris.shape[0]}")
classes = iris['species'].unique()
print(f"Classes de fleurs : {classes}")

# # 1. Filtrer les échantillons où petal_length > 4
filtered_iris = iris[iris['petal_length'] > 4]
print("\tÉchantillons où petal_length > 4 :\n\n")
print(filtered_iris.head())
print('_'*50)

# # 2. Ajouter une colonne petal_ratio
iris['petal_ratio'] = iris['petal_length'] / iris['petal_width']
print("\tDataFrame avec la colonne petal_ratio ajoutée :\n\n")
print(iris.head())
print('_'*50)

# # 3. Grouper les données par espèce et calculer les moyennes
grouped_means = iris.groupby('species').mean()
print("\tMoyennes par espèce :\n\n")
print(grouped_means)

    #Exercice 3 : Visualisation des données:


import matplotlib.pyplot as plt

sns.histplot(iris['sepal_length'], kde=True)
plt.title("Histogramme des longueurs de sépales")
plt.xlabel("Longueur de sépale (cm)")
plt.ylabel("Fréquence")
plt.show()

sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=iris)
plt.title("Nuage de points : Longueur de sépale vs Longueur de pétale")
plt.xlabel("Longueur de sépale")
plt.ylabel("Longueur de pétale")
plt.show()

correlation_matrix = iris.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show()

#   Exercice 4 : Nettoyage des données

import numpy as np

# Simuler des valeurs manquantes dans 'sepal_width'
np.random.seed(42)
iris['sepal_width'] = iris['sepal_width'].mask(np.random.random(len(iris)) < 0.1)
print(iris.isnull().sum()) 

mean_sepal_width = iris['sepal_width'].mean()
iris['sepal_width'].fillna(mean_sepal_width, inplace=True)
print(iris.isnull().sum()) 
