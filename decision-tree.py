import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset (Iris dataset for this example)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (labels)
feature_names = iris.feature_names
target_names = iris.target_names
#  Create a DataFrame for better data exploration
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['species'] = df['target'].map({0: target_names[0], 1: target_names[1], 2: target_names[2]})

# Data exploration 
# print(df.head())
# print(df.describe())
# print(df.groupby('species').mean())


# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (often helps with decision trees)
# if needed - it helps with certain datasets
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Creating the decision tree classifier with hyperparameters
clf = DecisionTreeClassifier(
    max_depth=3,  # Prevents overfitting
    min_samples_split=5,  # Minimum samples required to split a node
    criterion='entropy',  # Can use 'gini' or 'entropy'
    random_state=42
)

# Training the model
clf.fit(X_train, y_train)

# Cross-validation for more robust evaluation
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# predictions on the test set
y_pred = clf.predict(X_test)


