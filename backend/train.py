"""
train.py
--------
Master orchestration script for the Hospital Readmission Risk Model.

Phases executed
---------------
  Phase 3  -- Data validation
  Phase 4  -- Feature engineering comparison (with vs without)
  Phase 5  -- Leakage-safe data split (train 64% / cal 16% / test 20%)
  Phase 6  -- Baseline model comparison under 5-fold CV
  Phase 7a -- XGBoost Bayesian HPO via Optuna (100 trials, TPE sampler)
  Phase 7b -- LightGBM Bayesian HPO via Optuna (100 trials, TPE sampler)
  Phase 8  -- Early stopping evaluation
  Phase 9  -- Final ensemble assembly (XGB + LGBM soft-voting)
  Phase 10 -- Isotonic calibration
  Phase 11 -- MCC-maximising threshold (balanced on both classes)
  Phase 12 -- Business impact (gains + lift) on final test set
  Phase 13 -- SHAP explainability (XGB component of ensemble)
  Phase 15 -- Save artifacts: readmission_model_final.pkl + model_metadata.json
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

warnings.filterwarnings("ignore")

# -- Ensure project root is on sys.path --------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import apply_calibration, assess_calibration
from src.evaluation import (
    compute_gains_table,
    compute_metrics,
    evaluate_at_threshold,
    plot_cumulative_gains,
    plot_roc_curve,
    select_threshold_by_mcc,
    select_threshold_on_val,
)
from src.explainability import compute_global_importance, explain_patient
from src.features import engineer_features
from src.model import (
    RANDOM_STATE,
    build_ensemble_pipeline,
    build_lgbm_pipeline,
    build_pipeline,
    run_baseline_comparison,
    run_optuna_search,
)

# -- Paths -------------------------------------------------------------------
DATA_PATH      = PROJECT_ROOT / "dataset" / "hospital_readmissions.csv"
MODELS_DIR     = PROJECT_ROOT / "models"
CHARTS_DIR     = MODELS_DIR / "charts"
BASELINE_PKL   = MODELS_DIR / "readmission_model_baseline.pkl"
FINAL_PKL      = MODELS_DIR / "readmission_model_final.pkl"
METADATA_PATH  = MODELS_DIR / "model_metadata.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def validate_data(df: pd.DataFrame) -> None:
    section("PHASE 3 -- DATA VALIDATION")

    expected_cols = [
        "age", "time_in_hospital", "n_lab_procedures", "n_procedures",
        "n_medications", "n_outpatient", "n_inpatient", "n_emergency",
        "medical_specialty", "diag_1", "diag_2", "diag_3",
        "glucose_test", "A1Ctest", "change", "diabetes_med", "readmitted",
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"
    print("[OK] All expected columns present")

    assert df["readmitted"].isin(["yes", "no"]).all(), "Unexpected values in readmitted"
    dist = df["readmitted"].value_counts()
    print(f"[OK] Target distribution: {dist.to_dict()}")

    dupes = df.duplicated().sum()
    print(f"[OK] Duplicate rows: {dupes}")

    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"[OK] True null values: {total_nulls}")

    for col in ["medical_specialty", "diag_1", "diag_2", "diag_3"]:
        n = (df[col] == "Missing").sum()
        print(f"  Missing sentinel in {col}: {n} rows ({n/len(df)*100:.1f}%)")

    print(f"\n[OK] Dataset shape: {df.shape}")
    print("[OK] Data validation passed")


def compare_feature_sets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[bool, float]:
    section("PHASE 4 -- FEATURE ENGINEERING COMPARISON")
    print("Comparing: raw baseline features vs. raw + 14 engineered features")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe_with = build_pipeline(use_feature_engineering=True)
    pipe_without = build_pipeline(use_feature_engineering=False)

    scoring = {"roc_auc": "roc_auc", "avg_precision": "average_precision"}

    scores_with    = cross_validate(pipe_with,    X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    scores_without = cross_validate(pipe_without, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    print(f"\n  Without engineering: ROC-AUC = {scores_without['test_roc_auc'].mean():.4f} (+-{scores_without['test_roc_auc'].std():.4f})")
    print(f"  With 14 features:    ROC-AUC = {scores_with['test_roc_auc'].mean():.4f} (+-{scores_with['test_roc_auc'].std():.4f})")

    delta_auc = scores_with["test_roc_auc"].mean() - scores_without["test_roc_auc"].mean()
    print(f"\n  Delta ROC-AUC (with - without): {delta_auc:+.4f}")
    return True, delta_auc


def phase_baseline_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    section("PHASE 6 -- BASELINE MODEL COMPARISON")
    print("Running 5-fold CV for Dummy / LogReg / RandomForest / XGBoost...")

    comparison_df = run_baseline_comparison(X_train, y_train)
    print("\n" + comparison_df.to_string(index=False))

    best_model = comparison_df.iloc[0]["Model"]
    best_auc   = comparison_df.iloc[0]["Mean_ROC_AUC"]
    print(f"\n  Best: {best_model}  (ROC-AUC = {best_auc:.4f})")
    return comparison_df


def phase_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
) -> tuple[dict, dict]:
    section(f"PHASE 7 -- BAYESIAN HPO (Optuna TPE, {n_trials} trials each model)")
    print(f"  Tuning XGBoost  ({n_trials} trials)...")
    xgb_params, xgb_auc = run_optuna_search(X_train, y_train, n_trials=n_trials, model_type="xgb")

    print(f"\n  Tuning LightGBM ({n_trials} trials)...")
    lgbm_params, lgbm_auc = run_optuna_search(X_train, y_train, n_trials=n_trials, model_type="lgbm")

    print(f"\n  XGBoost  best CV ROC-AUC: {xgb_auc:.4f}")
    print(f"  LightGBM best CV ROC-AUC: {lgbm_auc:.4f}")

    return xgb_params, lgbm_params


def phase_early_stopping(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
) -> int:
    section("PHASE 8 -- EARLY STOPPING EVALUATION (XGB)")

    X_es_train, X_es_val, y_es_train, y_es_val = train_test_split(
        X_train, y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    es_params = best_params.copy()
    es_params["n_estimators"] = 1000
    es_params["early_stopping_rounds"] = 30

    from src.features import make_feature_transformer
    from src.preprocessing import build_preprocessor
    from xgboost import XGBClassifier

    feat_eng     = make_feature_transformer()
    preprocessor = build_preprocessor()

    X_es_train_eng = feat_eng.transform(X_es_train)
    X_es_val_eng   = feat_eng.transform(X_es_val)

    preprocessor.fit(X_es_train_eng, y_es_train)
    X_tr_proc  = preprocessor.transform(X_es_train_eng)
    X_val_proc = preprocessor.transform(X_es_val_eng)

    xgb_es = XGBClassifier(
        **es_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    xgb_es.fit(
        X_tr_proc, y_es_train,
        eval_set=[(X_val_proc, y_es_val)],
        verbose=False,
    )

    best_iteration = xgb_es.best_iteration
    print(f"  Early stopping best iteration: {best_iteration}")
    recommended_n = best_iteration + 1
    print(f"  -> Recommended n_estimators: {recommended_n}")
    return recommended_n


def main() -> None:
    section("HOSPITAL READMISSION RISK MODEL -- OPTIMIZED TRAINING PIPELINE v2")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Python:  {sys.version.split()[0]}")
    print("  Optimizations: Optuna TPE + XGB+LGBM Ensemble + Isotonic Cal + MCC Threshold")

    assert BASELINE_PKL.exists(), f"CRITICAL: Baseline model missing at {BASELINE_PKL}!"
    baseline_size = BASELINE_PKL.stat().st_size
    print(f"\n[OK] Baseline model intact: {BASELINE_PKL.name}  ({baseline_size:,} bytes)")

    # Load dataset
    section("LOADING DATASET")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")

    validate_data(df)

    X = df.drop(columns=["readmitted"]).copy()
    y = (df["readmitted"] == "yes").astype(int)

    # Phase 5: Leakage-safe split
    section("PHASE 5 -- LEAKAGE-SAFE DATA SPLIT")
    X_dev,  X_test,  y_dev,  y_test  = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=RANDOM_STATE
    )

    print(f"  Train (for CV):       {X_train.shape[0]} rows ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Calibration/Val:      {X_cal.shape[0]} rows ({X_cal.shape[0]/len(X)*100:.1f}%)")
    print(f"  Final Test (HOLDOUT): {X_test.shape[0]} rows ({X_test.shape[0]/len(X)*100:.1f}%)")

    use_engineering, delta_fe_auc = compare_feature_sets(X_train, y_train)

    comparison_df  = phase_baseline_comparison(X_train, y_train)
    xgb_row        = comparison_df[comparison_df["Model"] == "XGBoost (default)"]
    default_cv_auc = float(xgb_row["Mean_ROC_AUC"].iloc[0])

    xgb_params, lgbm_params = phase_optuna_tuning(X_train, y_train, n_trials=100)

    recommended_n_est = phase_early_stopping(X_train, y_train, xgb_params)
    tuned_n_est = xgb_params.get("n_estimators", 300)
    if abs(recommended_n_est - tuned_n_est) > 50:
        print(f"\n  Adjusting XGB n_estimators: {tuned_n_est} -> {recommended_n_est}")
        xgb_params["n_estimators"] = recommended_n_est

    # Assemble and fit ensemble
    section("PHASE 9 -- ENSEMBLE ASSEMBLY (XGBoost + LightGBM soft-voting)")
    ensemble = build_ensemble_pipeline(xgb_params=xgb_params, lgbm_params=lgbm_params)
    ensemble.fit(X_train, y_train)
    print("  [OK] Ensemble fitted on training set")

    from sklearn.model_selection import cross_val_score
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ensemble_cv_scores = cross_val_score(
        build_ensemble_pipeline(xgb_params=xgb_params, lgbm_params=lgbm_params),
        X_train, y_train, cv=cv5, scoring="roc_auc", n_jobs=1,
    )
    print(f"  Ensemble CV ROC-AUC: {ensemble_cv_scores.mean():.4f} (+-{ensemble_cv_scores.std():.4f})")
    print(f"  vs XGB-only default: {default_cv_auc:.4f}")

    # Phase 10: Calibration
    section("PHASE 10 -- CALIBRATION ASSESSMENT & APPLICATION")
    y_proba_cal_raw = ensemble.predict_proba(X_cal)[:, 1]
    cal_assess_raw  = assess_calibration(y_cal, y_proba_cal_raw, output_dir=CHARTS_DIR, n_bins=10)
    print(f"  Raw Brier score (on calibration set): {cal_assess_raw['brier_raw']:.4f}")

    calibrated_model = apply_calibration(ensemble, X_cal, y_cal, method="isotonic")
    y_proba_cal_iso  = calibrated_model.predict_proba(X_cal)[:, 1]

    brier_isotonic = brier_score_loss(y_cal, y_proba_cal_iso)
    print(f"  Isotonic Brier score:                 {brier_isotonic:.4f}")

    delta_brier = brier_isotonic - cal_assess_raw["brier_raw"]
    print(f"  Delta Brier (negative=improvement):   {delta_brier:+.4f}")

    if delta_brier < -0.002:
        print("  -> Calibration APPLIED (meaningful improvement)")
        inference_model     = calibrated_model
        calibration_applied = True
    else:
        print("  -> Using raw ensemble (calibration did not improve Brier)")
        inference_model     = ensemble
        calibration_applied = False
        brier_isotonic      = cal_assess_raw["brier_raw"]

    # Phase 11: Threshold selection
    section("PHASE 11 -- MCC THRESHOLD SELECTION (on calibration set)")
    y_proba_val = inference_model.predict_proba(X_cal)[:, 1]
    best_threshold, val_metrics = select_threshold_by_mcc(y_cal, y_proba_val)
    f2_threshold, f2_metrics    = select_threshold_on_val(y_cal, y_proba_val, beta=2.0)

    print(f"\n  [MCC] Selected threshold: {best_threshold:.4f}")
    print(f"  [MCC] Validation MCC:        {val_metrics.get('mcc', 0):.4f}")
    print(f"  [MCC] Validation precision:  {val_metrics['precision']:.4f}")
    print(f"  [MCC] Validation recall:     {val_metrics['recall']:.4f}")
    print(f"  [MCC] Balanced accuracy:     {val_metrics.get('balanced_accuracy', 0):.4f}")
    print(f"  [MCC] Flagged patients:      {val_metrics['flagged_pct']:.1f}%")

    # Phase 12: Final test set evaluation
    section("PHASE 12 -- FINAL TEST SET EVALUATION (one-shot)")
    y_proba_test = inference_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(y_test, y_proba_test, best_threshold, split_name="test")

    print(f"  Test ROC-AUC:        {test_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC:         {test_metrics['pr_auc']:.4f}")
    print(f"  Test Precision:      {test_metrics['precision']:.4f}")
    print(f"  Test Recall:         {test_metrics['recall']:.4f}")
    print(f"  Test F1:             {test_metrics['f1']:.4f}")
    print(f"  Test MCC:            {test_metrics.get('mcc', 0):.4f}")
    print(f"  Test Balanced Acc:   {test_metrics.get('balanced_accuracy', 0):.4f}")
    print(f"  Flagged:             {test_metrics['flagged_pct']:.1f}% of test patients")

    cm = test_metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    gains_table = compute_gains_table(y_test, y_proba_test)
    row30 = gains_table[gains_table["pct_patients"] == 30].iloc[0]
    print(f"\n  [***] TOP 30%: Capture rate {row30['capture_rate_pct']:.1f}%  Lift {row30['lift']:.3f}x")

    plot_cumulative_gains(gains_table, CHARTS_DIR, model_label="XGB+LGBM Ensemble v2")
    plot_roc_curve(y_test, y_proba_test, CHARTS_DIR)

    brier_test = brier_score_loss(y_test, y_proba_test)
    print(f"\n  Test Brier score: {brier_test:.4f}")

    # Phase 13: SHAP
    section("PHASE 13 -- SHAP EXPLAINABILITY (XGB component)")
    shap_success  = False
    importance_df = pd.DataFrame()

    try:
        xgb_sub_pipeline = ensemble.estimators_[0]
        sample_size = min(500, len(X_test))
        X_shap_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)

        importance_df = compute_global_importance(
            xgb_sub_pipeline, X_shap_sample,
            top_n=15, output_dir=CHARTS_DIR,
        )
        print(f"\n  Top 10 global features (mean |SHAP|):")
        print(importance_df.head(10).to_string(index=False))

        example_patient = X_test.iloc[[0]]
        explanation = explain_patient(xgb_sub_pipeline, example_patient)
        print(f"\n  Example patient raw risk score: {explanation['raw_risk_score']:.4f}")
        shap_success = True
    except Exception as e:
        print(f"  [!] SHAP computation failed: {e}")

    # Phase 15: Save artifacts
    section("PHASE 15 -- SAVING ARTIFACTS")
    joblib.dump(inference_model, FINAL_PKL)
    print(f"  [OK] Model saved: {FINAL_PKL}")

    metadata = {
        "artifact":              "readmission_model_final.pkl",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version":         "v2 (Optuna+Ensemble+MCC)",
        "python_version":        sys.version.split()[0],
        "pandas_version":        version("pandas"),
        "numpy_version":         version("numpy"),
        "scikit_learn_version":  version("scikit-learn"),
        "xgboost_version":       version("xgboost"),
        "joblib_version":        version("joblib"),
        "dataset": {
            "path":            str(DATA_PATH),
            "shape":           list(df.shape),
            "feature_count":   int(X.shape[1]),
            "target_column":   "readmitted",
            "target_encoding": {"yes": 1, "no": 0},
        },
        "split_strategy": {
            "method":           "stratified train_test_split",
            "train_pct":        64,
            "calibration_pct":  16,
            "test_pct":         20,
            "random_state":     RANDOM_STATE,
            "train_rows":       int(len(X_train)),
            "calibration_rows": int(len(X_cal)),
            "test_rows":        int(len(X_test)),
        },
        "feature_engineering": {
            "features_added": [
                "total_prior_visits", "procedures_per_day",
                "meds_per_day", "labs_per_day",
                "had_prior_inpatient", "had_prior_emergency",
                "lab_to_med_ratio", "utilisation_intensity",
                "is_high_utiliser", "meds_x_inpatient",
                "long_stay_flag", "no_test_flag",
                "diag_complexity", "specialty_x_inpatient",
            ],
            "total_engineered": 14,
            "leakage_safe": True,
            "inside_pipeline": True,
        },
        "preprocessing": {
            "numeric": "median imputation + StandardScaler",
            "age":     "mode imputation + OrdinalEncoder (6 ordered brackets)",
            "categorical": "mode imputation + OneHotEncoder(handle_unknown=ignore)",
            "missing_sentinel": "kept as category",
        },
        "baseline_comparison": comparison_df.to_dict(orient="records"),
        "hyperparameter_search": {
            "method":       "Optuna TPE (Bayesian)",
            "n_trials_xgb": 100,
            "n_trials_lgbm": 100,
            "cv_folds":     5,
            "scoring":      "roc_auc",
            "random_state": RANDOM_STATE,
            "xgb_best_params":  xgb_params,
            "lgbm_best_params": lgbm_params,
        },
        "early_stopping": {
            "recommended_n_estimators": recommended_n_est,
            "final_n_estimators":       xgb_params.get("n_estimators"),
        },
        "ensemble": {
            "type":    "soft-voting",
            "models":  ["XGBoost", "LightGBM"],
            "weights": [1.0, 1.0],
            "ensemble_cv_roc_auc": round(float(ensemble_cv_scores.mean()), 4),
        },
        "calibration": {
            "method":                 "isotonic (non-parametric)",
            "calibration_applied":    calibration_applied,
            "brier_raw_on_cal":       cal_assess_raw["brier_raw"],
            "brier_calibrated_on_cal": brier_isotonic,
            "delta_brier":            delta_brier,
            "brier_on_test":          brier_test,
        },
        "threshold": {
            "value":              best_threshold,
            "selection_method":   "MCC maximisation on calibration set",
            "val_precision":      val_metrics["precision"],
            "val_recall":         val_metrics["recall"],
            "val_f1":             val_metrics["f1"],
            "val_f2":             val_metrics.get("f2"),
            "val_mcc":            val_metrics.get("mcc"),
            "val_balanced_acc":   val_metrics.get("balanced_accuracy"),
            "val_flagged_pct":    val_metrics["flagged_pct"],
            "f2_threshold_comparison": {
                "value":     f2_threshold,
                "precision": f2_metrics["precision"],
                "recall":    f2_metrics["recall"],
                "flagged_pct": f2_metrics["flagged_pct"],
            },
        },
        "final_test_metrics": {
            "roc_auc":          test_metrics["roc_auc"],
            "pr_auc":           test_metrics["pr_auc"],
            "precision":        test_metrics["precision"],
            "recall":           test_metrics["recall"],
            "f1":               test_metrics["f1"],
            "f2":               test_metrics["f2"],
            "mcc":              test_metrics.get("mcc"),
            "balanced_accuracy":test_metrics.get("balanced_accuracy"),
            "flagged_pct":      test_metrics["flagged_pct"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "gains_table": gains_table.to_dict(orient="records"),
        "shap": {
            "implemented":  shap_success,
            "note":         "Computed on XGB sub-pipeline of ensemble",
            "top_features": importance_df.head(10).to_dict(orient="records")
            if shap_success and not importance_df.empty else [],
        },
        "charts_dir":              str(CHARTS_DIR),
        "baseline_model_preserved": str(BASELINE_PKL),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, default=str)
    print(f"  [OK] Metadata saved: {METADATA_PATH}")

    section("TRAINING COMPLETE -- SUMMARY")
    print(f"  Model:               XGB + LGBM Soft-Voting Ensemble")
    print(f"  Final Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    print(f"  Final Test PR-AUC:   {test_metrics['pr_auc']:.4f}")
    print(f"  Final Test MCC:      {test_metrics.get('mcc', 0):.4f}")
    print(f"  Final Balanced Acc:  {test_metrics.get('balanced_accuracy', 0):.4f}")
    print(f"  Operating threshold: {best_threshold:.4f}  (MCC-selected on val set)")
    print(f"  Calibration:         {'Isotonic (applied)' if calibration_applied else 'Not applied'}")
    print(f"  Top-30% capture:     {row30['capture_rate_pct']:.1f}%  (lift={row30['lift']:.3f}x)")


if __name__ == "__main__":
    main()
