import pandas as pd
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = r'C:\Users\patala durga prasad\Desktop\machine_learning_project\archive (1)\list_bbox_celeba.xlsx'
data = pd.read_excel(file_path)

# Check the data types of columns
print(data.dtypes)

# Identify numeric columns
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

# Exclude non-numeric columns and 'height'
X = data.drop(columns=numeric_columns)
y = data['height']

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=4)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save the changes to a new Excel file
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['height'] = y_resampled

new_file_path = r'C:\Users\patala durga prasad\Desktop\machine_learning_project\resampled_list_bbox_celeba.xlsx'
resampled_data.to_excel(new_file_path, index=False)
