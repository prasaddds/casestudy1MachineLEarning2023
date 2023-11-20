# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:15:05 2023

@author: patala durga prasad
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

file_path = r'C:\Users\patala durga prasad\Desktop\machine_learning_project\archive (1)\celeba_encoded_outliers_handled.xlsx'
data = pd.read_excel(file_path)

features = data.drop(columns=['image_id', 'Young_-1.0', 'Young_1.0'])

kmeans = KMeans(n_clusters=5, n_init=10)
data['Cluster'] = kmeans.fit_predict(features)

plt.figure(figsize=(8, 6))

# Scatter plot to visualize clusters (considering the first two columns as an example)
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=data['Cluster'], cmap='viridis', marker='o', edgecolor='black')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()
