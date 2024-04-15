import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def load_data(file_path):
    """
    Load dataset from file.
    
    Args:
    file_path (str): Path to the dataset file.

    Returns:
    data (DataFrame): Entire dataset.
    """
    # Load the cleaned dataset
    data = pd.read_csv(file_path)
    
    return data


def load_split_data(file_path, target_column='y', test_size=0.2, random_state=None):
    """
    Load dataset from file, split it into features and labels,
    and further split it into training and testing sets.
    
    Args:
    file_path (str): Path to the dataset file.
    target_column (str): Name of the target column.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int or None): Random state for reproducibility.

    Returns:
    X_train (DataFrame): Features of the training set.
    X_test (DataFrame): Features of the testing set.
    y_train (Series): Labels of the training set.
    y_test (Series): Labels of the testing set.
    """
    # Load the cleaned dataset
    data = pd.read_csv(file_path)
    
    # Split the data into features and labels
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    # Initialize and train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)

    print('######################')
    print('Model Performance')
    print('######################')

    print("Accuracy:", accuracy)
    print("AUC-ROC:", auc_roc)
    print("Classification Report:")
    print(report)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:', conf_matrix)

    # Extract values from the confusion matrix
    true_negative, false_positive, false_negative, true_positive = conf_matrix.ravel()

    # Print the counts
    print("True Positive Count:", true_positive)
    print("False Positive Count:", false_positive)
    print("True Negative Count:", true_negative)
    print("False Negative Count:", false_negative)

    # For Client
    loss_one = false_positive * 100
    loss_zero = false_negative * 40  # Adjusted for class 0 mispredictions
    losses = loss_one + loss_zero

    print('######################')
    print('Total Losses')
    print('######################')

    print('Total loss from mispredictions of class 1: $', loss_one)
    print('Total loss from mispredictions of class 0: $', loss_zero)
    print('Total loss from mispredictions: $', losses)


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot the feature importances of a given model.

    Args:
    model: A trained model with feature_importances_ attribute.
    feature_names (list): A list of feature names.
    top_n (int): Number of top features to display.

    Returns:
    None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(top_n), importances[indices], color="b", align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, top_n])
    plt.tight_layout()
    plt.show()
