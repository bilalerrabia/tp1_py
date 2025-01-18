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

# ----------- Exercice 1: Data Loading and Preparation -----------

# Load the dataset
data = pd.read_csv('moteurs_structured.csv')

# Display the first few rows of the dataset
print("Dataset preview:\n", data.head())

# Encode the target column 'Type de moteur'
label_encoder = LabelEncoder()
data['Type_encoded'] = label_encoder.fit_transform(data['Type de moteur'])
print("\nEncoded motor types:\n", data[['Type de moteur', 'Type_encoded']])

# Separate features (X) and target (y)
X = data[['Résistance (ohms)', 'Inductance (mH)', 'Vitesse nominale (RPM)', 'Couple maximal (Nm)']]
y = data['Type_encoded']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Scale the features (important for k-NN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display a summary of the dataset
print("\nDataset statistics:\n", data.describe())

# ----------- Exercice 2: k-NN Model Implementation -----------

# Find the best k using cross-validation
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

# Re-train the k-NN model with the best k
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

# ----------- Exercice 3: Model Evaluation -----------

# Predict the test set results
y_pred = knn_model.predict(X_test_scaled)

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Plot the confusion matrix for better visualization
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Display classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ----------- Exercice 4: Visualization -----------

# Visualize data by reducing features to two dimensions
X_reduced = X[['Résistance (ohms)', 'Couple maximal (Nm)']]

# Scatter plot with different colors for each class
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced['Résistance (ohms)'], X_reduced['Couple maximal (Nm)'], c=y, cmap='viridis', edgecolor='k')
plt.title("Motor Data Visualization")
plt.xlabel("Résistance (ohms)")
plt.ylabel("Couple maximal (Nm)")
plt.colorbar(label="Motor Type")
plt.show()

# Highlight misclassified examples
incorrect_indices = y_test != y_pred
plt.figure(figsize=(8, 6))
plt.scatter(X_test['Résistance (ohms)'], X_test['Couple maximal (Nm)'], c=y_test, cmap='viridis', edgecolor='k', label='Correct')
plt.scatter(X_test[incorrect_indices]['Résistance (ohms)'], X_test[incorrect_indices]['Couple maximal (Nm)'], c='red', label='Misclassified')
plt.title("Misclassified Points")
plt.xlabel("Résistance (ohms)")
plt.ylabel("Couple maximal (Nm)")
plt.legend()
plt.show()

# ----------- Exercice 5: Linear Regression Comparison -----------

# Create and train a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predict using the regression model
y_pred_linear = linear_model.predict(X_test_scaled)

# Evaluate the linear regression model
mse = mean_squared_error(y_test, y_pred_linear)
r2 = r2_score(y_test, y_pred_linear)

print(f"\nLinear Regression Mean Squared Error (MSE): {mse}")
print(f"Linear Regression Coefficient of Determination (R²): {r2}")


