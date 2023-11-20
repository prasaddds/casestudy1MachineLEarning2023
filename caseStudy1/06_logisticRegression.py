# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:39:44 2023

@author: patala durga prasad
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
file_path = r'C:\Users\patala durga prasad\Desktop\machine_learning_project\archive (1)\celeba_encoded_outliers_handled.xlsx'
data = pd.read_excel(file_path)

# Define features (attributes) and target variable
features = data.drop(columns=['image_id', 'Young_-1.0', 'Young_1.0'])  # Excluding 'image_id' and target columns
target = data['Young_-1.0']  # Replace with the correct target column


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define hyperparameters to tune
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # Range of values for 'C'

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_C = grid_search.best_params_['C']

# Create the final Logistic Regression model with the best hyperparameters
final_model = LogisticRegression(C=best_C)
final_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = final_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f"Best 'C' value found: {best_C}")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
