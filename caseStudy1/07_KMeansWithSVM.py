# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:48:42 2023

@author: patala durga prasad
"""

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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

# Define the cluster labels as new features
features_with_clusters = pd.concat([features, pd.get_dummies(data['Cluster'], prefix='Cluster')], axis=1)

# Assuming 'target' is one of 'Young_-1.0' or 'Young_1.0' for classification
target = data['Young_-1.0']  # Replace with the correct target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_with_clusters, target, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm = SVC()

# Fit SVM on the training data
svm.fit(X_train, y_train)

# Make predictions on the test set
predictions = svm.predict(X_test)

# Evaluate the SVM classifier
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

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
