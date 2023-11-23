import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data using make_blobs
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create KNeighborsClassifier instances for k=5 and k=1
knn5 = KNeighborsClassifier(n_neighbors=5)
knn1 = KNeighborsClassifier(n_neighbors=1)

# Fit the models on the training data
knn5.fit(X_train, y_train)
knn1.fit(X_train, y_train)

# Make predictions
y_pred_5 = knn5.predict(X_test)
y_pred_1 = knn1.predict(X_test)

# Calculate and print accuracy for k=5 and k=1
print("Accuracy with k=5:", accuracy_score(y_test, y_pred_5) * 100)
print("Accuracy with k=1:", accuracy_score(y_test, y_pred_1) * 100)

# Create a scatter plot to visualize the predictions
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_5, marker='^', s=100, edgecolors='blue')
plt.title("Predicted values with k=5", fontsize=20)
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_1, marker='^', s=100, edgecolors='violet')
plt.title("Predicted values with k=1", fontsize=20)
plt.show()
