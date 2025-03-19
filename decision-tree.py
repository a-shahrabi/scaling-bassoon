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

# Data exploration (uncomment to see)
# print(df.head())
# print(df.describe())
# print(df.groupby('species').mean())
