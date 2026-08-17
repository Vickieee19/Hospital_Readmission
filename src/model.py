"""
model.py
--------
Pipeline construction, baseline model comparison, and hyperparameter
optimization for the hospital readmission risk model.

Contents
--------
build_pipeline()              Full sklearn Pipeline: feature eng -> preprocess -> XGB
build_lgbm_pipeline()         Full sklearn Pipeline: feature eng -> preprocess -> LGBM
build_ensemble_pipeline()     Soft-voting ensemble of XGB + LGBM pipelines
run_baseline_comparison()     Dummy / LogReg / RF / XGBoost under 5-fold CV
run_optuna_search()           Bayesian HPO via Optuna TPE (100 trials, ROC-AUC)
run_hyperparameter_search()   Legacy RandomizedSearchCV (kept as fallback)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import make_feature_transformer
from src.preprocessing import build_preprocessor

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

RANDOM_STATE = 42
N_SPLITS     = 5


# â”€â”€ Pipeline builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_pipeline(
    xgb_params: dict[str, Any] | None = None,
    use_feature_engineering: bool = True,
) -> Pipeline:
    """
    Build the complete end-to-end sklearn Pipeline with XGBoost classifier.

    If use_feature_engineering=True (default):
        raw DataFrame
            v FunctionTransformer  (14 engineered features)
            v ColumnTransformer    (impute + scale + encode)
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


def build_lgbm_pipeline(
    lgbm_params: dict[str, Any] | None = None,
) -> Pipeline:
    """
    Build the complete end-to-end sklearn Pipeline with LightGBM classifier.

    LightGBM uses leaf-wise growth (vs XGB's depth-wise), so it makes
    different types of errors â€” makes it complementary in an ensemble.

    Parameters
    ----------
    lgbm_params : dict, optional
        Override default LightGBM parameters.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    from lightgbm import LGBMClassifier

    default_lgbm = {
        "n_estimators":     300,
        "max_depth":        -1,          # uncapped; controlled by num_leaves
        "num_leaves":       31,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_samples":20,
        "reg_alpha":        0.0,
        "reg_lambda":       1.0,
        "objective":        "binary",
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
        "verbosity":        -1,
    }
    if lgbm_params:
        default_lgbm.update(lgbm_params)

    lgbm = LGBMClassifier(**default_lgbm)

    return Pipeline([
        ("feature_eng",  make_feature_transformer()),
        ("preprocessor", build_preprocessor()),
        ("classifier",   lgbm),
    ])


def build_ensemble_pipeline(
    xgb_params: dict[str, Any] | None = None,
    lgbm_params: dict[str, Any] | None = None,
    xgb_weight: float = 1.0,
    lgbm_weight: float = 1.0,
) -> Pipeline:
    """
    Build a soft-voting ensemble of XGBoost and LightGBM pipelines.

    Each pipeline is a full feature-eng â†’ preprocess â†’ classifier stack.
    The VotingClassifier averages their predicted probabilities (soft voting),
    which is more powerful than hard voting for well-calibrated models.

    Parameters
    ----------
    xgb_params  : XGBoost hyperparameters (post-Optuna)
    lgbm_params : LightGBM hyperparameters (post-Optuna)
    xgb_weight  : weight for XGBoost predictions (default 1.0)
    lgbm_weight : weight for LightGBM predictions (default 1.0)

    Returns
    -------
    sklearn.pipeline.Pipeline wrapping a VotingClassifier
    """
    from lightgbm import LGBMClassifier

    # Build full params
    xgb_full = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0,
    }
    if xgb_params:
        xgb_full.update(xgb_params)

    lgbm_full = {
        "n_estimators": 300, "max_depth": -1, "num_leaves": 31,
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_samples": 20, "reg_alpha": 0.0, "reg_lambda": 1.0,
        "objective": "binary", "random_state": RANDOM_STATE,
        "n_jobs": -1, "verbosity": -1,
    }
    if lgbm_params:
        lgbm_full.update(lgbm_params)

    # Each constituent pipeline is independently built with feature engineering
    xgb_pipe = Pipeline([
        ("feature_eng",  make_feature_transformer()),
        ("preprocessor", build_preprocessor()),
        ("classifier",   XGBClassifier(**xgb_full)),
    ])

    lgbm_pipe = Pipeline([
        ("feature_eng",  make_feature_transformer()),
        ("preprocessor", build_preprocessor()),
        ("classifier",   LGBMClassifier(**lgbm_full)),
    ])

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb_pipe), ("lgbm", lgbm_pipe)],
        voting="soft",
        weights=[xgb_weight, lgbm_weight],
        n_jobs=1,   # each model already parallelises internally
    )

    return ensemble


# â”€â”€ Baseline comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
                n_jobs=1,
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


# â”€â”€ Optuna Bayesian Hyperparameter Search (PRIMARY) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    model_type: str = "xgb",   # "xgb" or "lgbm"
) -> tuple[dict[str, Any], float]:
    """
    Bayesian hyperparameter optimisation using Optuna TPE sampler.

    Why Optuna over RandomizedSearchCV?
    ------------------------------------
    - TPE (Tree-structured Parzen Estimator) learns which parameter regions
      are promising from previous trials â€” smarter than random sampling
    - 100 informed trials >> 40 random trials for finding the true optimum
    - Supports pruning of unpromising trials for speed
    - Returns full trial history for analysis

    Parameters
    ----------
    X_train    : raw training features
    y_train    : binary target
    n_trials   : number of Optuna trials (default 100)
    model_type : "xgb" for XGBoost, "lgbm" for LightGBM

    Returns
    -------
    (best_params, best_cv_auc)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def xgb_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        }
        pipeline = build_pipeline(xgb_params=params)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=1,
        )
        return float(scores.mean())

    def lgbm_objective(trial: optuna.Trial) -> float:
        from lightgbm import LGBMClassifier
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 600),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        }
        pipeline = build_lgbm_pipeline(lgbm_params=params)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=1,
        )
        return float(scores.mean())

    objective = xgb_objective if model_type == "xgb" else lgbm_objective
    sampler    = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study      = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params   = study.best_trial.params
    best_cv_auc   = study.best_trial.value

    print(f"\n[Optuna/{model_type.upper()}] Best CV ROC-AUC : {best_cv_auc:.4f}")
    print(f"[Optuna/{model_type.upper()}] Best params     : {best_params}")

    return best_params, best_cv_auc


# â”€â”€ Legacy RandomizedSearchCV (kept as fallback) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 40,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    RandomizedSearchCV over XGBoost hyperparameters (legacy fallback).

    Prefer run_optuna_search() for better results. This is kept for
    backward compatibility.

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
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
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

