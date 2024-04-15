import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

from Functions import train_evaluate_model, load_split_data


# Load and split the data
X_train, X_test, y_train, y_test = load_split_data('Data/PPDataDNA.csv', target_column='y', test_size=0.2, random_state=42)

# Define a list of models to iterate over
models = [
    GradientBoostingClassifier(),
    LogisticRegression(n_jobs=-1, max_iter=1000),
    SVC(),
    XGBClassifier(n_jobs=-1),
    RandomForestClassifier(n_jobs=-1)
]

# Loop over each model
for model in models:
    print(f"Training and evaluating model: {model.__class__.__name__}")
    train_evaluate_model(model, X_train, X_test, y_train, y_test)
    print("=" * 50)
