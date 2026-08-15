"""
model.py
--------
Pipeline construction, baseline model comparison, and hyperparameter
optimization for the hospital readmission risk model.

Contents
--------
build_pipeline()            Full sklearn Pipeline: feature eng -> preprocess -> XGB
run_baseline_comparison()   Dummy / LogReg / RF / XGBoost under 5-fold CV
run_hyperparameter_search() RandomizedSearchCV on XGBoost, 40 iterations
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import make_feature_transformer
from src.preprocessing import build_preprocessor

# ── Constants ──────────────────────────────────────────────────────────────

RANDOM_STATE = 42
N_SPLITS     = 5


# ── Pipeline builder ───────────────────────────────────────────────────────

def build_pipeline(
    xgb_params: dict[str, Any] | None = None,
    use_feature_engineering: bool = True,
) -> Pipeline:
    """
    Build the complete end-to-end sklearn Pipeline.

    If use_feature_engineering=True (default):
        raw DataFrame
            v FunctionTransformer  (6 engineered features)
            v ColumnTransformer    (impute + scale + encode)
            v XGBClassifier

    If use_feature_engineering=False:
        raw DataFrame
            v ColumnTransformer    (impute + scale + encode, raw cols only)
            v XGBClassifier

    Parameters
    ----------
    xgb_params : dict, optional
        Override default XGBoost parameters.
    use_feature_engineering : bool
        Whether to include the FunctionTransformer step.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    default_xgb = {
        "n_estimators":     300,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
        "verbosity":        0,
    }
    if xgb_params:
        default_xgb.update(xgb_params)

    xgb = XGBClassifier(**default_xgb)

    if use_feature_engineering:
        return Pipeline([
            ("feature_eng",  make_feature_transformer()),
            ("preprocessor", build_preprocessor()),
            ("classifier",   xgb),
        ])
    else:
        # Build preprocessor with RAW features only (no engineered columns)
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
        from src.features import AGE_ORDER

        raw_numeric = [
            "time_in_hospital", "n_lab_procedures", "n_procedures",
            "n_medications", "n_outpatient", "n_inpatient", "n_emergency",
        ]
        preprocessor_raw = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scl", StandardScaler()),
                ]), raw_numeric),
                ("age", Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ord", OrdinalEncoder(
                        categories=[AGE_ORDER],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    )),
                ]), ["age"]),
                ("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), [
                    "medical_specialty", "diag_1", "diag_2", "diag_3",
                    "glucose_test", "A1Ctest", "change", "diabetes_med",
                ]),
            ],
            remainder="drop",
        )
        return Pipeline([
            ("preprocessor", preprocessor_raw),
            ("classifier",   xgb),
        ])


# ── Baseline comparison ────────────────────────────────────────────────────

def run_baseline_comparison(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Compare four models under identical 5-fold stratified CV.

    Models
    ------
    Dummy         : stratified random baseline
    LogisticReg   : linear benchmark
    RandomForest  : tree ensemble benchmark
    XGBoost       : final algorithm candidate

    Each model uses the SAME feature engineering + preprocessing pipeline
    (from build_pipeline or an equivalent inner pipeline) so the comparison
    is apples-to-apples.

    Returns
    -------
    pd.DataFrame  columns: Model, Mean_ROC_AUC, Std_ROC_AUC, Mean_PR_AUC,
                           Std_PR_AUC, Notes
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)

    scoring = {"roc_auc": "roc_auc", "avg_precision": "average_precision"}

    feat_eng   = make_feature_transformer()
    preprocess = build_preprocessor()

    candidates = {
        "Dummy (stratified)": Pipeline([
            ("feature_eng",  make_feature_transformer()),
            ("preprocessor", build_preprocessor()),
            ("classifier",   DummyClassifier(
                strategy="stratified", random_state=RANDOM_STATE)),
        ]),
        "Logistic Regression": Pipeline([
            ("feature_eng",  make_feature_transformer()),
            ("preprocessor", build_preprocessor()),
            ("classifier",   LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Random Forest": Pipeline([
            ("feature_eng",  make_feature_transformer()),
            ("preprocessor", build_preprocessor()),
            ("classifier",   RandomForestClassifier(
                n_estimators=200, max_depth=6,
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "XGBoost (default)": build_pipeline(),
    }

    rows = []
    for name, pipeline in candidates.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                pipeline, X, y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,        # inner XGB already uses n_jobs=-1
                return_train_score=False,
            )

        rows.append({
            "Model":         name,
            "Mean_ROC_AUC":  round(scores["test_roc_auc"].mean(),        4),
            "Std_ROC_AUC":   round(scores["test_roc_auc"].std(),         4),
            "Mean_PR_AUC":   round(scores["test_avg_precision"].mean(),  4),
            "Std_PR_AUC":    round(scores["test_avg_precision"].std(),   4),
            "Notes":         "",
        })

    df = pd.DataFrame(rows).sort_values("Mean_ROC_AUC", ascending=False)
    df = df.reset_index(drop=True)
    return df


# ── Hyperparameter search ──────────────────────────────────────────────────

def run_hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 40,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    RandomizedSearchCV over XGBoost hyperparameters.

    Search space chosen to cover regularisation, tree structure, and
    stochasticity — wide enough to escape the hand-tuned defaults but
    not so large that it takes hours.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (raw, before feature engineering).
    y_train : pd.Series
        Binary target (0/1).
    n_iter : int
        Number of parameter settings sampled (default 40).

    Returns
    -------
    (best_pipeline, best_params)
        best_pipeline : fitted Pipeline with best found params
        best_params   : dict of classifier__ prefixed params
    """
    param_dist = {
        "classifier__n_estimators":     [100, 200, 300, 400, 500],
        "classifier__max_depth":        [3, 4, 5, 6],
        "classifier__learning_rate":    [0.01, 0.03, 0.05, 0.07, 0.1],
        "classifier__min_child_weight": [1, 3, 5, 7],
        "classifier__subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__gamma":            [0, 0.1, 0.2, 0.5, 1.0],
        "classifier__reg_alpha":        [0, 0.01, 0.1, 0.5, 1.0],
        "classifier__reg_lambda":       [0.5, 1.0, 1.5, 2.0, 5.0],
    }

    base_pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        refit=True,           # refit best params on full X_train
        random_state=RANDOM_STATE,
        n_jobs=1,             # XGB already parallelises internally
        verbose=1,
        return_train_score=False,
    )

    search.fit(X_train, y_train)

    best_params = {
        k.replace("classifier__", ""): v
        for k, v in search.best_params_.items()
    }

    print(f"\n[HyperOpt] Best CV ROC-AUC : {search.best_score_:.4f}")
    print(f"[HyperOpt] Best params     : {best_params}")

    return search.best_estimator_, best_params
