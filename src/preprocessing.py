"""
preprocessing.py
----------------
Builds the sklearn ColumnTransformer that handles:

  1. Numeric features (raw + engineered)  -> median imputation -> StandardScaler
  2. Age ordinal feature                  -> mode imputation  -> OrdinalEncoder
  3. Categorical features (OHE)           -> mode imputation  -> OneHotEncoder

Design notes
------------
- The "Missing" string in medical_specialty is intentionally kept as a
  category (49.5 % of rows). It carries genuine signal that the specialty
  was not recorded, likely related to ED/unscheduled admissions. Imputing
  it with the mode ("InternalMedicine") would destroy this signal.

- OrdinalEncoder uses handle_unknown="use_encoded_value" + unknown_value=-1
  so unseen age brackets at inference time are mapped to a safe sentinel.

- OneHotEncoder uses handle_unknown="ignore" so unseen categories at
  inference time produce an all-zero row without raising an error.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.features import (
    AGE_ORDER,
    ALL_NUMERIC,
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Return a fitted-ready ColumnTransformer for the readmission pipeline.

    Column mapping
    --------------
    'num'  : ALL_NUMERIC (raw + engineered numeric columns)
    'age'  : ORDINAL_FEATURES (['age'])
    'cat'  : CATEGORICAL_FEATURES (8 OHE columns)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    age_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ordinal",
            OrdinalEncoder(
                categories=[AGE_ORDER],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,    ALL_NUMERIC),
            ("age", age_pipeline,        ORDINAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor
