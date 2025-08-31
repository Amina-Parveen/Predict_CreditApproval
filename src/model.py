# src/model.py
import joblib
import pandas as pd

# Function to load trained model
def load_model(path="/Users/mac/Documents/Predict_CreditApproval/models/logistic_credit_model.joblib"):
    """
    Load the trained Logistic Regression model.
    """
    return joblib.load(path)

# Function to make predictions
def predict(model, X):
    """
    Make predictions using the trained model.
    X should be a pandas DataFrame or numpy array.
    """
    return model.predict(X)

# Optional: function to evaluate predictions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(y_true, y_pred):
    """
    Evaluate model predictions and return metrics as a dictionary.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
