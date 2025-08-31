# main.py
"""
Main script to run the Credit Card Approval prediction pipeline.
"""

import pandas as pd
from src.model import load_model, predict, evaluate

# Step 1: Load processed dataset
data_path = "/Users/mac/Documents/Predict_CreditApproval/data//processed/credit_approval_processed.csv"
df = pd.read_csv(data_path)

# Step 2: Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# Step 3: Load trained model
model = load_model("/Users/mac/Documents/Predict_CreditApproval/models/logistic_credit_model.joblib")

# Step 4: Make predictions
y_pred = predict(model, X)

# Step 5: Evaluate predictions
metrics = evaluate(y, y_pred)

# Step 6: Print results
print("\n🔹 Model Evaluation on Full Dataset:")
for key, value in metrics.items():
    print(f"{key.capitalize()}: {value:.4f}")
