import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Functions import train_evaluate_model

# Load the cleaned dataset
data = pd.read_csv('Data/PPDataDNA.csv')

# Split the data into features and labels
X = data.drop(columns=['y'])
y = data['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()

# Train and evaluate the model using the function
train_evaluate_model(model, X_train, X_test, y_train, y_test)