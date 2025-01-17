# TP 1 : Introduction à Python et manipulation des données


## Objectifs pédagogiques

À la fin de ce TP, nous serons capable de :
1. Charger et inspecter un dataset avec **Pandas**.
2. Manipuler les données : filtrer, transformer, grouper et analyser.
3. Visualiser des données avec **Matplotlib** et **Seaborn**.
4. Identifier et traiter les valeurs manquantes dans un dataset.

## Prérequis

### Connaissances nécessaires
- Bases de Python (boucles, fonctions, manipulations de listes).
- Comprendre le concept de DataFrame.

### Matériel nécessaire

```bash
pip install numpy pandas matplotlib seaborn
   ![Texte alternatif](1.jpg)
   Exercices:

Exercice 1 : Chargement et inspection des données:

  1.Charger le dataset Iris avec Seaborn :

  ![Texte alternatif](2.jpg)

  2.Explorer les données avec head(), info() et describe() :

  ![Texte alternatif](3.jpg)

  3.Identifier le nombre d échantillons et les classes de fleurs :

  ![Texte alternatif](4.jpg)

Exercice 2 : Exploration des données:

  1.Filtrer les échantillons où petal_length > 4 :
  2.Ajouter une colonne petal_ratio :
  3.Grouper les données par espèce et calculer les moyennes :

  ![Texte alternatif](5.jpg)

Exercice 3 : Visualisation des données

  1.Histogramme des longueurs de sépales :
  2.Nuage de points coloré par espèce :
  3.Matrice de corrélation avec une carte thermique :

  ![Texte alternatif](6.jpg)

Exercice 4 : Nettoyage des données

  1.Simuler des valeurs manquantes dans sepal_width :
  2.Remplacer ces valeurs par la moyenne :

  ![Texte alternatif](7.jpg)


5.Résultats attendus

  1.Dataset nettoyé :

Aucune valeur manquante dans sepal_width.

Colonne petal_ratio ajoutée.

  2.Graphiques :

Histogramme montrant la distribution des longueurs de sépales.

Nuage de points illustrant la relation entre sepal_length et petal_length, coloré par espèce.

Matrice de corrélation montrant les relations entre les variables numériques.
  ![Texte alternatif](8.1.jpg)
  ![Texte alternatif](8.2.jpg)



  3.Ajoutez des titres, des légendes et des annotations pour rendre les graphiques compréhensibles:

  ![Texte alternatif](9.jpg)


## AUTEUR:
  BILAL ERRABIA