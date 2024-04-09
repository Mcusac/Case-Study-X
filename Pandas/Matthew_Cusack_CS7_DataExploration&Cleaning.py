import pandas as pd

# Load data using pandas
data = pd.read_csv('Data/final_project(5).csv')

# Display basic information about the dataset
print("Shape of the dataset:", data.shape)
print("Data types of each column:")
print(data.dtypes)

# Remove '%' from column 'x32' and convert to float
data['x32'] = data['x32'].str.replace('%', '').astype(float) * 100

# Remove '$' from column 'x37' and convert to float
data['x37'] = data['x37'].str.replace('$', '').astype(float)

# Display the first 3 lines of all categorical columns
categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
for col in categorical_columns:
    print(f"\nFirst 3 lines of column '{col}':")
    print(data[col].head(3))

# One-hot encode categorical columns
data = pd.get_dummies(data)

# Display basic information about the preprocessed dataset
print("\nShape of the preprocessed dataset before dropping NaN rows:", data.shape)

# Drop rows with missing values (NaN)
data.dropna(inplace=True)

# Display basic information about the preprocessed dataset after dropping NaN rows
print("\nShape of the preprocessed dataset after dropping NaN rows:", data.shape)

# Save the preprocessed dataset as a new CSV file
data.to_csv('Data/PPDataDNA.csv', index=False)
