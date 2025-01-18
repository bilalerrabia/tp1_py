# README for k-Nearest Neighbors (k-NN) and Linear Regression Project

## Project Overview
This project implements a machine learning workflow to classify different types of electric motors using the **k-Nearest Neighbors (k-NN)** algorithm and compares its performance with **linear regression**. The dataset contains motor parameters such as resistance, inductance, nominal speed, and maximum torque, as well as their corresponding motor types.

## Dataset
The dataset file is named `moteurs_structured.csv` and contains the following columns:
- **Type de moteur**: The type of motor (e.g., "DC", "Asynchrone", "Pas-à-pas").
- **Résistance (ohms)**: Electrical resistance in ohms.
- **Inductance (mH)**: Electrical inductance in millihenries.
- **Vitesse nominale (RPM)**: Nominal speed in revolutions per minute.
- **Couple maximal (Nm)**: Maximum torque in newton-meters.

## Project Workflow

### 1. Data Loading and Preparation
- Load the dataset using `pandas`.
- Encode the motor types into numerical values using `LabelEncoder`.
- Split the data into training (80%) and testing (20%) sets.
- Features: `Résistance (ohms)`, `Inductance (mH)`, `Vitesse nominale (RPM)`, `Couple maximal (Nm)`.
- Target: Encoded values of `Type de moteur`.

### 2. k-NN Model Implementation
- Create a k-NN classifier with `k=3` using `Scikit-learn`.
- Train the model on the training set.

### 3. Model Evaluation
- Predict motor types using the test set.
- Generate a confusion matrix to evaluate performance.
- Compute precision, recall, and F1-score for classification performance.

### 4. Visualization
- Visualize the data by reducing features to two dimensions (`Résistance` and `Couple maximal`).
- Plot the data points with colors representing different classes.
- Highlight misclassified points in red.

### 5. Linear Regression Comparison
- Train a linear regression model using the same training data.
- Predict using the test data and evaluate performance using:
  - Mean Squared Error (MSE).
  - Coefficient of determination (R²).
- Compare the results of linear regression with k-NN.

## Prerequisites
Ensure the following Python libraries are installed:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

Install them using:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## How to Run the Code
1. Place the `moteurs_structured.csv` file in the same directory as the Python script.
2. Run the Python script using:
   ```bash
   python tp3_code.py
   ```
3. The script will:
   - Display evaluation metrics for the k-NN model.
   - Visualize the classification results.
   - Print MSE and R² for the linear regression model.

## Expected Outputs
1. **Confusion Matrix**: Shows the distribution of true vs predicted labels.
2. **Classification Report**: Includes precision, recall, and F1-score.
3. **Visualization**: A scatter plot showing data points and misclassified examples.
4. **Regression Metrics**: MSE and R² values for linear regression.

## Notes
- Ensure the column names in the dataset match exactly as listed above.
- The k-NN algorithm is more suitable for this classification task than linear regression.
- Experiment with different values of `k` to see how the model performance changes.

## Author
This project was developed as part of a machine learning tutorial focusing on supervised classification and comparison with regression techniques.

