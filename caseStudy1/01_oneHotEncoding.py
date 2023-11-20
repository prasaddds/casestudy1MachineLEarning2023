import pandas as pd

# File path to the list_attr_celeba dataset in Excel format
file_path = r'C:\Users\patala durga prasad\Desktop\machine_learning_project\archive (1)\list_attr_celeba.xlsx'

# Read the dataset into a pandas DataFrame from Excel
celeba_data = pd.read_excel(file_path)

# Column to perform one hot encoding on
target_column = 'Young'

# Perform one hot encoding using pandas get_dummies function
one_hot_encoded = pd.get_dummies(celeba_data[target_column], prefix=target_column)

# Concatenate the one hot encoded columns with the original dataset
celeba_data_encoded = pd.concat([celeba_data, one_hot_encoded], axis=1)

# Drop the original column if needed
celeba_data_encoded.drop(columns=[target_column], inplace=True)

# Save the encoded dataset to a new Excel file
output_file_path = r'C:\Users\patala durga prasad\Desktop\machine_learning_project\archive (1)\celeba_encoded.xlsx'
celeba_data_encoded.to_excel(output_file_path, index=False)
