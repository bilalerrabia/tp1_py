# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Exercice 1:


data = pd.read_csv('moteurs_structured.csv')

print("Dataset preview:\n", data.head())

label_encoder = LabelEncoder()
data['Type_encoded'] = label_encoder.fit_transform(data['Type de moteur'])
print("\nEncoded motor types:\n", data[['Type de moteur', 'Type_encoded']])

X = data[['Résistance (ohms)', 'Inductance (mH)', 'Vitesse nominale (RPM)', 'Couple maximal (Nm)']]
y = data['Type_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nDataset statistics:\n", data.describe())

# Exercice 2: 

best_k = 0
best_score = 0

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='f1_weighted')  # Use weighted F1-score
    mean_score = scores.mean()
    print(f"k={k}, F1-score={mean_score:.4f}")
    if mean_score > best_score:
        best_k = k
        best_score = mean_score

print(f"\nBest k: {best_k}, with F1-score: {best_score:.4f}")

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

# Exercice 3:

y_pred = knn_model.predict(X_test_scaled)

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Exercice 4: 

X_reduced = X[['Résistance (ohms)', 'Couple maximal (Nm)']]

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced['Résistance (ohms)'], X_reduced['Couple maximal (Nm)'], c=y, cmap='viridis', edgecolor='k')
plt.title("Motor Data Visualization")
plt.xlabel("Résistance (ohms)")
plt.ylabel("Couple maximal (Nm)")
plt.colorbar(label="Motor Type")
plt.show()

incorrect_indices = y_test != y_pred
plt.figure(figsize=(8, 6))
plt.scatter(X_test['Résistance (ohms)'], X_test['Couple maximal (Nm)'], c=y_test, cmap='viridis', edgecolor='k', label='Correct')
plt.scatter(X_test[incorrect_indices]['Résistance (ohms)'], X_test[incorrect_indices]['Couple maximal (Nm)'], c='red', 
            label='Misclassified')

plt.title("Misclassified Points")
plt.xlabel("Résistance (ohms)")
plt.ylabel("Couple maximal (Nm)")
plt.legend()
plt.show()

# Exercice 5:

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

y_pred_linear = linear_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred_linear)
r2 = r2_score(y_test, y_pred_linear)

print(f"\nLinear Regression Mean Squared Error (MSE): {mse}")
print(f"Linear Regression Coefficient of Determination (R²): {r2}")


