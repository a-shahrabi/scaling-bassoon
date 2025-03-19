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

