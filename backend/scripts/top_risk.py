import pandas as pd
import joblib

from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("hospital_readmissions.csv")

# Features and target
X = df.drop(columns=["readmitted"]).copy()
y = (df["readmitted"] == "yes").astype(int)

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Load trained model
model = joblib.load("readmission_model.pkl")

# Predict readmission probability
y_proba = model.predict_proba(X_test)[:, 1]

# Combine predictions with actual results
results = pd.DataFrame({
    "probability": y_proba,
    "actual": y_test.to_numpy()
})

# Highest-risk patients first
results = results.sort_values(
    "probability",
    ascending=False
).reset_index(drop=True)

# Total actual readmissions
total_readmissions = results["actual"].sum()

print("Total test patients:", len(results))
print("Total actual readmissions:", total_readmissions)
print()

# Top-risk capture
for percent in [10, 20, 30, 40, 50]:

    n_patients = int(len(results) * percent / 100)

    top_patients = results.iloc[:n_patients]

    captured = top_patients["actual"].sum()

    capture_rate = captured / total_readmissions

    print(
        f"Top {percent}% | "
        f"Patients: {n_patients} | "
        f"Readmissions captured: {captured} | "
        f"Capture rate: {capture_rate * 100:.1f}%"
    )