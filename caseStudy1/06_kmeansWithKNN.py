# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:46:13 2023

@author: patala durga prasad
"""

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = r'C:\Users\patala durga prasad\Desktop\machine_learning_project\archive (1)\celeba_encoded_outliers_handled.xlsx'
data = pd.read_excel(file_path)

# Assuming 'features' are all columns except 'image_id', 'Young_-1.0', and 'Young_1.0'
features = data.drop(columns=['image_id', 'Young_-1.0', 'Young_1.0'])

# Initialize K-Means for clustering into, say, 5 clusters
kmeans = KMeans(n_clusters=5,n_init=10)
data['Cluster'] = kmeans.fit_predict(features)

# Assuming 'target' is one of 'Young_-1.0' or 'Young_1.0' for classification
target = data['Young_-1.0']  # Replace with the correct target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Define hyperparameters to tune
param_grid = {'n_neighbors': [3, 5, 7, 9]}  # Example values for 'n_neighbors'

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_n_neighbors = grid_search.best_params_['n_neighbors']

# Create the final KNN classifier with the best hyperparameters
final_knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
final_knn.fit(X_train, y_train)

# Make predictions on the test set
predictions = final_knn.predict(X_test)

# Evaluate the KNN classifier
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f"Best 'n_neighbors' value found: {best_n_neighbors}")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
