import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

from Functions import train_evaluate_model, load_split_data, plot_feature_importance, load_data


# Load data using load_data function
data_file = 'Data/PPDataDNA.csv'
data = load_data(data_file)

# Load and split the data
X_train, X_test, y_train, y_test = load_split_data(data_file, target_column='y', test_size=0.2, random_state=42)

# initializer the models
RFC_model = RandomForestClassifier(n_jobs=-1)
XGBC_model = XGBClassifier(n_jobs=-1)

# Load best parameters from the JSON file
with open('best_params.json', 'r') as f:
    best_params = json.load(f)

# Initialize models with best parameters
best_XGBC_model = XGBClassifier(**best_params['XGBC'])
best_RFC_model = RandomForestClassifier(**best_params['RFC'])

# Create a list of the best models
models = [best_XGBC_model, best_RFC_model]

# Loop over each model
for model in models:
    print(f"Training and evaluating model: {model.__class__.__name__}")
    train_evaluate_model(model, X_train, X_test, y_train, y_test)
    print("=" * 50)

# Plot feature importance
feature_names = data.columns.tolist()
if 'y' in feature_names:
    feature_names.remove('y')
plot_feature_importance(best_XGBC_model, feature_names)
plot_feature_importance(best_RFC_model, feature_names)

