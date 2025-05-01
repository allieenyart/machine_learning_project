from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load our dataset
df = pd.read_csv("./data/data_banknote_authentication.csv")
X, y = df.drop(columns=['Classifier']), df['Classifier']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an ID3 decision tree classifier (uses entropy as the splitting criterion) for using k-fold
id3_tree_cv = DecisionTreeClassifier(criterion="entropy", random_state=42)

# Use StratifiedKFold to preserve class balance in each fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validated accuracy
cv_accuracy = cross_val_score(id3_tree_cv, X, y, cv=kf, scoring='accuracy')
cv_f1 = cross_val_score(id3_tree_cv, X, y, cv=kf, scoring='f1')
cv_auc = cross_val_score(id3_tree_cv, X, y, cv=kf, scoring='roc_auc')

print(f"Cross-validated Accuracy: {cv_accuracy.mean():.2f} +/- {cv_accuracy.std():.2f}")
print(f"Cross-validated F1 Score: {cv_f1.mean():.2f} +/- {cv_f1.std():.2f}")
print(f"Cross-validated ROC-AUC: {cv_auc.mean():.2f} +/- {cv_auc.std():.2f}")

# New tree for regular ID3 splitting
id3_tree = DecisionTreeClassifier(criterion="entropy", random_state=42)

# Train the model
id3_tree.fit(X_train, y_train)

# Make predictions
y_pred = id3_tree.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Evaluate f1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Genuine", "Fake"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print(classification_report(y_test, y_pred))

# Get the feature importances for feature importance graph
importances = id3_tree.feature_importances_
feature_names = X.columns

plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("ID3 Feature Importance")
plt.show()

# Plot of tree
plt.figure(figsize=(12, 8))
plt.title("Decision Tree Visualization")
plot_tree(
    id3_tree,
    feature_names=X.columns,
    class_names=["Genuine", "Fake"],
    filled=True,
    max_depth=3,
    impurity=False,    
    rounded=True
)
plt.show()
