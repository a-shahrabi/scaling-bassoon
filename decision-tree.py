import numpy as np
# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# Step 1: Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Step 2: Load a dataset (Iris dataset for this example)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (labels)

# Step 3: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Step 5: Train the model
clf.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = clf.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 8: Visualize the decision tree
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
