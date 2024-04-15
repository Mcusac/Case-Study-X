import pandas as pd
import json
import numpy as np
import matplotlib as plt
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

## Tune Model w/ GridSearchCV (replace param_grid with your desired parameters)
XGBC_param_grid = {
    'n_jobs': [-1],
    'missing': [-999, np.nan],
    'silent': [True, False],
    'nthread': [-1, 1, 4] 
                   }

RFC_param_grid = {
    'n_jobs': [5],
    #'n_estimators': [100, 1000, 2000],
    #'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth':[None, 5, 10, 15],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf':[1, 5, 10],
    #'max_features':["sqrt", "log2"],
    #'max_leaf_nodes':[None, 10, 20, 30],
    #'min_impurity_decrease':[0.0, 0.1, 0.2],
                  }

XGBC_grid_search = GridSearchCV(XGBC_model, XGBC_param_grid, cv=5, n_jobs=20)
RFC_grid_search = GridSearchCV(RFC_model, RFC_param_grid, cv=5, n_jobs=10)

## Train models
XGBC_grid_search.fit(X_train, y_train)
print("XGBC gridsearch done")
RFC_grid_search.fit(X_train, y_train)
print("RFC gridsearch done")

## Save & Show Best Parameters
print("XGBC Best Parameters:", XGBC_grid_search.best_params_)
print("RFC Best Parameters:", RFC_grid_search.best_params_)

# Access the best parameters
XGBC_best_params = XGBC_grid_search.best_params_
RFC_best_params = RFC_grid_search.best_params_

# Create a dictionary for combined storage
best_params = {
    'XGBC': XGBC_best_params,
    'RFC': RFC_best_params
}

# Save to a JSON file
with open('best_params.json', 'w') as f:
    json.dump(best_params, f)

with open('best_params.json', 'r') as f:
    loaded_params = json.load(f)