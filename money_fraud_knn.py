from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

# Load our dataset
df = pd.read_csv("./data/data_banknote_authentication.csv")
X, y = df.drop(columns=['Classifier']), df['Classifier']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Evaluate f1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_auc:.2f}")

y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))

scores = cross_val_score(knn, X, y, cv=5)
print(f"Mean CV accuracy: {scores.mean():.4f}")

X_test_noisy = X_test + np.random.normal(0, 0.1, X_test.shape)
print("Noisy accuracy:", knn.score(X_test_noisy, y_test))

#Visualization
# Reduce to 2D
X_2D = PCA(n_components=2).fit_transform(X)

# Train on 2D version
knn_2d = KNeighborsClassifier(n_neighbors=20)
knn_2d.fit(X_2D, y)

# Plot decision boundary
h = 0.02
x_min, x_max = X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1
y_min, y_max = X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.4)
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap='bwr', edgecolor='k', s=20)
plt.title("KNN Decision Boundary (PCA-Reduced)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()