<!-- ======================================= -->
<!-- README.md for Predict Credit Card Approvals -->
<!-- ======================================= -->

<h1 align="center">💳 Predict Credit Card Approvals</h1>

<p align="center">
  A machine learning project to automatically predict credit card approval using Logistic Regression and hyperparameter optimization.
</p>

---

## 📌 Project Overview

Credit card approval is a critical task for banks and financial institutions. This project builds an **automatic predictor** to classify applicants as **approved** or **denied** based on historical data.  

Key highlights:

- Handle **missing values**  
- Encode **categorical features**  
- Scale numerical features for **consistent modeling**  
- Train **Logistic Regression** with **GridSearchCV** for hyperparameter tuning  
- Evaluate performance using **accuracy, precision, recall, and F1-score**

---

## 📂 Project Structure

```
Predict_CreditApproval/
├─ data/
│ └─ processed/                   # Cleaned dataset for modeling
├─ notebooks/
│ ├─ 01_data_cleaning.ipynb       # Data cleaning and preprocessing
│ └─ 02_modeling.ipynb            # Model training and evaluation
├─ src/
│ ├─ model.py                     # Functions to load, predict, and evaluate model
├─ models/
│ └─ logistic_credit_model.joblib # Trained Logistic Regression model
├─ main.py                        # Run the full pipeline
├─ requirements.txt               # Project dependencies
└─ README.md
```
---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Predict_CreditApproval.git
cd Predict_CreditApproval
(Optional) Create a virtual environment:
# Using conda
conda create -n credit_approval python=3.10
conda activate credit_approval

# Or using virtualenv
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
Install dependencies:
pip install -r requirements.txt
```
---

## 🧹 Data Preprocessing
The project uses the Credit Approval dataset from UCI repository.
**Steps performed in 01_data_cleaning.ipynb:**

- Rename columns to human-readable names
- Handle missing values (median for numerical, mode for categorical)
- Encode categorical features and target variable (class)
- Scale features using StandardScaler
- Save cleaned dataset to data/processed/credit_approval_processed.csv

---

## 🤖 Model Training

- Implemented Logistic Regression with hyperparameter tuning via GridSearchCV.
- Performed train-test split.
- Evaluated model using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - Trained model is saved as:
**models/logistic_credit_model.joblib**

---

## ⚡ Running the Pipeline

Run the full pipeline using:
```
python main.py
```
This will:
- Load the processed dataset
- Load the trained model
- Make predictions
- Print evaluation metrics

---

## 📊 Model Evaluation
Sample metrics on the dataset:
**Metric	Score**
- Accuracy	0.90
- Precision	0.92
- Recall	0.88
- F1 Score	0.90
(Values may vary slightly depending on train-test split)

---

## 📝 Notes
SMOTE (data balancing) is optional and not included to avoid compatibility issues.  
You can extend the project with other ML models or deploy it via Flask/FastAPI.  
Original raw dataset is not included in the repo; it is fetched using ucimlrepo in the notebook.  

---

## 🔗 References
**UCI Credit Approval DatasetDataset:** [UCI Credit Approval Dataset](https://archive.ics.uci.edu/ml/datasets/credit+approval)  
**scikit-learn Documentation:** [scikit-learn Documentation](https://scikit-learn.org/stable/)  
**imbalanced-learn Documentation:** [imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)  

---

##  Author

👩 **Amina Parveen**  
📧 **Contact:** [LinkedIn](https://www.linkedin.com/in/amina-parveen-9606182a2)  
🌐 **GitHub:** [@Amina-Parveen](https://github.com/Amina-Parveen)  

---

✨ *This project is for learning and academic purposes only.*
