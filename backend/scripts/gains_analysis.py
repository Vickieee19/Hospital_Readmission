import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


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

results = pd.DataFrame({
    "probability": y_proba,
    "actual": y_test.to_numpy()
})

results = results.sort_values(
    "probability",
    ascending=False
).reset_index(drop=True)

total_readmissions = results["actual"].sum()

percentages = [10, 20, 30, 40, 50]
capture_rates = []

for percent in percentages:

    n_patients = int(len(results) * percent / 100)

    top_patients = results.iloc[:n_patients]

    captured = top_patients["actual"].sum()

    capture_rate = (captured / total_readmissions) * 100

    capture_rates.append(capture_rate)

    print(
        f"Top {percent}% | "
        f"Readmissions captured: {captured} | "
        f"Capture rate: {capture_rate:.1f}%"
    )


# Create gains chart
plt.figure(figsize=(8, 5))

plt.plot(
    percentages,
    capture_rates,
    marker="o",
    linewidth=2,
    label="XGBoost"
)

# Random targeting baseline
plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    label="Random targeting"
)

plt.xlabel("Percentage of Patients Prioritized")
plt.ylabel("Percentage of Readmissions Captured")
plt.title("Readmission Gains Analysis")

plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()