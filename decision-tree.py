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


# Evaluating the model with multiple metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# Visualize the confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# visualizing the confusion matrix
plot_confusion_matrix(cm, target_names)

# Visualizing the decision tree
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=feature_names, class_names=target_names, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
