import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


df = pd.read_csv("hospital_readmissions.csv")

X = df.drop(columns=["readmitted"]).copy()
y = (df["readmitted"] == "yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

model = joblib.load("readmission_model.pkl")

y_proba = model.predict_proba(X_test)[:, 1]


thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

print("Threshold Analysis\n")

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    flagged = y_pred.sum()
    flagged_percent = flagged / len(y_pred) * 100

    print(
        f"Threshold: {threshold:.2f} | "
        f"Flagged: {flagged_percent:.1f}% | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )