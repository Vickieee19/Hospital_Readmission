import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("hospital_readmissions.csv")


# Features and target
X = df.drop(columns=["readmitted"]).copy()
y = (df["readmitted"] == "yes").astype(int)


# Feature groups
numeric_features = [
    "time_in_hospital",
    "n_lab_procedures",
    "n_procedures",
    "n_medications",
    "n_outpatient",
    "n_inpatient",
    "n_emergency"
]

ordinal_features = ["age"]

categorical_features = [
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "glucose_test",
    "A1Ctest",
    "change",
    "diabetes_med"
]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# Age order
age_order = [
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)"
]


# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Age preprocessing
age_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(
        categories=[age_order],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])


# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("age", age_pipeline, ordinal_features),
    ("cat", categorical_pipeline, categorical_features)
])


# XGBoost classifier
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


# Complete pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])


# Train
model.fit(X_train, y_train)


# Predict readmission probability
y_proba = model.predict_proba(X_test)[:, 1]

print("First 10 predicted probabilities:")
print(y_proba[:10])


joblib.dump(model, "readmission_model.pkl")

print("Model saved successfully.")