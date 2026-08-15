import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# Load dataset
df = pd.read_csv("hospital_readmissions.csv")


# Features and target
X = df.drop(columns=["readmitted"]).copy()
y = (df["readmitted"] == "yes").astype(int)


# Recreate the same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# Load trained model
model = joblib.load("readmission_model.pkl")


# Predict probability of readmission
y_proba = model.predict_proba(X_test)[:, 1]


# ROC-AUC
auc = roc_auc_score(y_test, y_proba)

print("ROC-AUC:", round(auc, 4))


# Decision threshold
threshold = 0.35

y_pred = (y_proba >= threshold).astype(int)

print("Threshold:", threshold)


# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))