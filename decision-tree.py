import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['species'] = df['target'].map({0: target_names[0], 1: target_names[1], 2: target_names[2]})

plt.figure(figsize=(12, 10))
sns.pairplot(df, hue='species', markers=['o', 's', 'D'], plot_kws={'alpha': 0.7})
plt.suptitle('Pairwise Relationships Between Features', y=1.02)
plt.show()

plt.figure(figsize=(15, 10))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    for species in target_names:
        sns.kdeplot(df[df['species'] == species][feature], label=species)
    plt.title(f'Distribution of {feature} by Species')
    plt.xlabel(feature)
    plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = df.iloc[:, :4].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                          param_grid=param_grid, 
                          cv=5, 
                          scoring='accuracy',
                          verbose=1,
                          n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

best_clf = grid_search.best_estimator_

clf = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=5,
    criterion='entropy',
    random_state=42
)

clf.fit(X_train, y_train)

cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(cm, target_names)

plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=feature_names, class_names=target_names, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
perm_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': result.importances_mean,
    'Std': result.importances_std
}).sort_values('Importance', ascending=False)

print("\nPermutation Importance (more reliable):")
print(perm_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=perm_importance, palette='viridis')
plt.title('Permutation Feature Importance')
plt.errorbar(x=perm_importance['Importance'], y=range(len(perm_importance)), 
             xerr=perm_importance['Std'], fmt='none', color='black', capsize=3)
plt.tight_layout()
plt.show()

train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.15)
plt.plot(train_sizes, test_mean, label='Cross-validation score', color='green', marker='s')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='green', alpha=0.15)
plt.title('Learning Curves')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
for i, class_name in enumerate(target_names):
    y_prob = y_pred_proba[:, i]
    y_test_bin = (y_test == i).astype(int)
    
    fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

max_depths = range(1, 20)
train_accuracy = []
test_accuracy = []

for depth in max_depths:
    temp_clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    temp_clf.fit(X_train, y_train)
    
    train_accuracy.append(accuracy_score(y_train, temp_clf.predict(X_train)))
    test_accuracy.append(accuracy_score(y_test, temp_clf.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_accuracy, 'o-', label='Training Accuracy', color='blue')
plt.plot(max_depths, test_accuracy, 's-', label='Testing Accuracy', color='red')
plt.axvline(x=clf.max_depth, color='green', linestyle='--', label=f'Current model depth ({clf.max_depth})')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Maximum Depth vs Accuracy')
plt.legend()
plt.grid(True)
plt.show()