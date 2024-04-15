import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load data using numpy
data = np.genfromtxt('final_project(5).csv', delimiter=',', dtype=None, encoding=None, names=True)

# Determine the type of data in each column
data_types = data.dtype

# Print the data type of each column
print("Data types of each column:")
print(data_types)

# Extract labels (last column)
target_index = -1
labels = data[data.dtype.names[target_index]].astype(int)

# Remove the label column from the data
data = np.delete(data, target_index, axis=1)

# Convert categorical columns to one-hot encoding
categorical_columns = [i for i, dtype in enumerate(data.dtype) if np.issubdtype(dtype, np.str_)]
encoder = OneHotEncoder(sparse=False, dtype=np.int, categories='auto')
one_hot_encoded = encoder.fit_transform(data[:, categorical_columns])
data = np.delete(data, categorical_columns, axis=1)

# Concatenate the one-hot encoded data with the numerical data
data = np.concatenate([data.astype(float), one_hot_encoded], axis=1)

# Display basic information about the dataset
print("Shape of the dataset:", data.shape)
print("Number of labels:", len(labels))

# Print the first 5 rows and first 5 columns of the data
print("First 5 rows and first 5 columns of the data:")
print(data[:5, :5])

# Check for missing values in the data
missing_values = np.isnan(data)

# Count the number of missing values in each column
missing_counts = np.sum(missing_values, axis=0)

# Display the number of missing values in each column
print("Number of missing values in each column:")
for i, count in enumerate(missing_counts):
    print(f"Column {i}: {count}")
