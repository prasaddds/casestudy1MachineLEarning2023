import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# File path to the preprocessed dataset
file_path = r'C:\Users\patala durga prasad\Desktop\machine_learning_project\archive (1)\celeba_encoded_outliers_handled.xlsx'

# Read the dataset into a pandas DataFrame from Excel
celeba_data = pd.read_excel(file_path)

# Define features (independent variables) and target variable
features = celeba_data.drop(columns=['Young_1.0', 'image_id'])  # Exclude 'Young_1.0' and 'image_id' columns from features
target = celeba_data['Young_1.0']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the Lasso Regression model
lasso = Lasso()

# Define the hyperparameters to tune
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model on the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_alpha = grid_search.best_params_['alpha']

# Use the best hyperparameters to create the final model
final_model = Lasso(alpha=best_alpha)
final_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = final_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Best Alpha: {best_alpha}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Plotting actual vs. predicted values with regression line
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, predictions, 1))(np.unique(y_test)), color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values with Regression Line (Lasso Regression)')
plt.show()
