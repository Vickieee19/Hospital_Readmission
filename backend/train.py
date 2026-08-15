"""
train.py
--------
Master orchestration script for the Hospital Readmission Risk Model.

Run from the project root:
    .venv\\Scripts\\python.exe backend/train.py

Phases executed
---------------
  Phase 3  — Data validation
  Phase 4  — Feature engineering comparison (with vs without)
  Phase 5  — Leakage-safe data split (train 64% / cal 16% / test 20%)
  Phase 6  — Baseline model comparison under 5-fold CV
  Phase 7  — XGBoost hyperparameter optimisation (RandomizedSearchCV)
  Phase 8  — Early stopping evaluation
  Phase 9  — Final pipeline assembly
  Phase 10 — Probability calibration assessment + optional application
  Phase 11 — Threshold selection on calibration/dev data (F2-maximising)
  Phase 12 — Business impact (gains + lift) on final test set
  Phase 13 — SHAP explainability
  Phase 15 — Save artifacts: readmission_model_final.pkl + model_metadata.json

Artifacts saved to
------------------
  backend/models/readmission_model_final.pkl
  backend/models/model_metadata.json
  backend/models/charts/
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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

warnings.filterwarnings("ignore")

# ── Ensure project root is on sys.path ──────────────────────────────────────
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
    select_threshold_on_val,
)
from src.explainability import compute_global_importance, explain_patient
from src.features import engineer_features
from src.model import (
    RANDOM_STATE,
    build_pipeline,
    run_baseline_comparison,
    run_hyperparameter_search,
)

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_PATH      = PROJECT_ROOT / "dataset" / "hospital_readmissions.csv"
MODELS_DIR     = PROJECT_ROOT / "models"
CHARTS_DIR     = MODELS_DIR / "charts"
BASELINE_PKL   = MODELS_DIR / "readmission_model_baseline.pkl"
FINAL_PKL      = MODELS_DIR / "readmission_model_final.pkl"
METADATA_PATH  = MODELS_DIR / "model_metadata.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def banner(msg: str) -> None:
    print(f"\n── {msg}")


# ────────────────────────────────────────────────────────────────────────────
# PHASE 3 — DATA VALIDATION
# ────────────────────────────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> None:
    section("PHASE 3 — DATA VALIDATION")

    # Expected columns
    expected_cols = [
        "age", "time_in_hospital", "n_lab_procedures", "n_procedures",
        "n_medications", "n_outpatient", "n_inpatient", "n_emergency",
        "medical_specialty", "diag_1", "diag_2", "diag_3",
        "glucose_test", "A1Ctest", "change", "diabetes_med", "readmitted",
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"
    print("[OK] All expected columns present")

    # Target
    assert df["readmitted"].isin(["yes", "no"]).all(), \
        "Unexpected values in 'readmitted'"
    dist = df["readmitted"].value_counts()
    print(f"[OK] Target distribution: {dist.to_dict()}")

    # Duplicates
    dupes = df.duplicated().sum()
    print(f"[OK] Duplicate rows: {dupes}  (dataset unchanged)")

    # Null values (true nulls)
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"[OK] True null values: {total_nulls}")

    # Sentinel 'Missing' categories
    for col in ["medical_specialty", "diag_1", "diag_2", "diag_3"]:
        n = (df[col] == "Missing").sum()
        print(f"  'Missing' sentinel in {col}: {n} rows ({n/len(df)*100:.1f}%)")
    print("  -> Decision: 'Missing' kept as a category (carries clinical signal)")

    # Numeric range checks
    numeric_cols = [
        "time_in_hospital", "n_lab_procedures", "n_procedures",
        "n_medications", "n_outpatient", "n_inpatient", "n_emergency",
    ]
    for col in numeric_cols:
        neg = (df[col] < 0).sum()
        if neg > 0:
            print(f"  [!] {col}: {neg} negative values (investigate)")
        else:
            print(f"  [OK] {col}: no negative values")

    # Age categories
    valid_ages = {
        "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    }
    unknown_ages = set(df["age"].unique()) - valid_ages
    if unknown_ages:
        print(f"  [!] Unknown age brackets: {unknown_ages}")
    else:
        print("  [OK] Age brackets: all valid")

    print(f"\n[OK] Dataset shape: {df.shape}")
    print("[OK] Data validation passed — no silent modifications made")


# ────────────────────────────────────────────────────────────────────────────
# PHASE 4 — FEATURE ENGINEERING COMPARISON
# ────────────────────────────────────────────────────────────────────────────

def compare_feature_sets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    section("PHASE 4 — FEATURE ENGINEERING COMPARISON")
    print("Comparing: raw baseline features vs. raw + engineered features")
    print("(5-fold CV, roc_auc scoring, same XGBoost defaults)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Pipeline WITH engineering (default build_pipeline)
    pipe_with = build_pipeline()

    # Pipeline WITHOUT engineering — use same structure but with a passthrough FE
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import Pipeline
    from src.preprocessing import build_preprocessor

    raw_features_only = [
        "time_in_hospital", "n_lab_procedures", "n_procedures", "n_medications",
        "n_outpatient", "n_inpatient", "n_emergency",
    ]
    ordinal_only   = ["age"]
    cat_only       = [
        "medical_specialty", "diag_1", "diag_2", "diag_3",
        "glucose_test", "A1Ctest", "change", "diabetes_med",
    ]

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
    from src.features import AGE_ORDER

    preprocessor_no_eng = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
            ]), raw_features_only),
            ("age", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(
                    categories=[AGE_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]), ordinal_only),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_only),
        ],
        remainder="drop",
    )

    from xgboost import XGBClassifier
    pipe_without = Pipeline([
        ("preprocessor", preprocessor_no_eng),
        ("classifier",   XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )),
    ])

    scoring = {"roc_auc": "roc_auc", "avg_precision": "average_precision"}

    scores_with = cross_validate(
        pipe_with, X_train, y_train,
        cv=cv, scoring=scoring, n_jobs=1,
    )
    scores_without = cross_validate(
        pipe_without, X_train, y_train,
        cv=cv, scoring=scoring, n_jobs=1,
    )

    print(f"\n  Without engineering: "
          f"ROC-AUC = {scores_without['test_roc_auc'].mean():.4f} "
          f"(±{scores_without['test_roc_auc'].std():.4f})  "
          f"PR-AUC = {scores_without['test_avg_precision'].mean():.4f}")
    print(f"  With engineering:    "
          f"ROC-AUC = {scores_with['test_roc_auc'].mean():.4f} "
          f"(±{scores_with['test_roc_auc'].std():.4f})  "
          f"PR-AUC = {scores_with['test_avg_precision'].mean():.4f}")

    delta_auc = (scores_with['test_roc_auc'].mean()
                 - scores_without['test_roc_auc'].mean())
    print(f"\n  Delta ROC-AUC (with - without): {delta_auc:+.4f}")

    if delta_auc >= 0.002:
        use_engineering = True
        print("  -> Decision: KEEP engineered features (meaningful improvement)")
    elif delta_auc >= 0:
        use_engineering = True
        print("  -> Decision: KEEP engineered features (negligible but non-harmful)")
    else:
        use_engineering = False
        print("  -> Decision: DISCARD engineered features (performance degraded)")

    return use_engineering, delta_auc


# ────────────────────────────────────────────────────────────────────────────
# PHASE 6 — BASELINE MODEL COMPARISON
# ────────────────────────────────────────────────────────────────────────────

def phase_baseline_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    section("PHASE 6 — BASELINE MODEL COMPARISON")
    print("Running 5-fold CV for Dummy / LogReg / RandomForest / XGBoost …")

    comparison_df = run_baseline_comparison(X_train, y_train)

    print("\n" + comparison_df.to_string(index=False))

    best_model = comparison_df.iloc[0]["Model"]
    best_auc   = comparison_df.iloc[0]["Mean_ROC_AUC"]
    print(f"\n  Best: {best_model}  (ROC-AUC = {best_auc:.4f})")

    return comparison_df


# ────────────────────────────────────────────────────────────────────────────
# PHASE 7 — HYPERPARAMETER OPTIMISATION
# ────────────────────────────────────────────────────────────────────────────

def phase_hyperopt(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    default_cv_auc: float,
) -> tuple:
    section("PHASE 7 — HYPERPARAMETER OPTIMISATION (RandomizedSearchCV)")
    print(f"  Baseline CV ROC-AUC: {default_cv_auc:.4f}")
    print("  Search space: 9 hyperparameters, 40 iterations, 5-fold CV …\n")

    best_pipeline, best_params = run_hyperparameter_search(
        X_train, y_train, n_iter=40
    )

    # Cross-validate the best pipeline to get a fair estimate
    from sklearn.model_selection import cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # Re-build a fresh pipeline with best params (search.best_estimator_ is
    # already fitted on full X_train — just report its CV score from search)
    print(f"\n  Default CV ROC-AUC : {default_cv_auc:.4f}")
    print(f"  Optimised CV ROC-AUC: (see RandomizedSearchCV output above)")

    return best_pipeline, best_params


# ────────────────────────────────────────────────────────────────────────────
# PHASE 8 — EARLY STOPPING
# ────────────────────────────────────────────────────────────────────────────

def phase_early_stopping(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
) -> int:
    section("PHASE 8 — EARLY STOPPING EVALUATION")

    # Use 10% of training data as an early-stopping eval set
    X_es_train, X_es_val, y_es_train, y_es_val = train_test_split(
        X_train, y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    es_params = best_params.copy()
    # Build pipeline with higher n_estimators to let early stopping find the right number
    es_params["n_estimators"] = 1000
    es_params["early_stopping_rounds"] = 30

    from src.features import make_feature_transformer, engineer_features
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
        **{k: v for k, v in es_params.items()},
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
    print(f"  (out of max 1000; original best_params n_estimators={best_params.get('n_estimators', 300)})")

    recommended_n = best_iteration + 1
    print(f"  -> Recommended n_estimators: {recommended_n}")
    return recommended_n


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    section("HOSPITAL READMISSION RISK MODEL — TRAINING PIPELINE")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Python:  {sys.version.split()[0]}")

    # ── Verify baseline is intact ──────────────────────────────────────────
    assert BASELINE_PKL.exists(), \
        f"CRITICAL: Baseline model missing at {BASELINE_PKL}!"
    baseline_size = BASELINE_PKL.stat().st_size
    print(f"\n[OK] Baseline model intact: {BASELINE_PKL.name}  ({baseline_size:,} bytes)")

    # ── Load dataset ───────────────────────────────────────────────────────
    section("LOADING DATASET")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")

    # ── Phase 3: Data validation ───────────────────────────────────────────
    validate_data(df)

    # ── Prepare features and target ────────────────────────────────────────
    X = df.drop(columns=["readmitted"]).copy()
    y = (df["readmitted"] == "yes").astype(int)

    # ── Phase 5: Leakage-safe data split ──────────────────────────────────
    section("PHASE 5 — LEAKAGE-SAFE DATA SPLIT")
    #
    # Strategy:
    #   80% DEVELOPMENT  -> used for CV, tuning, calibration, threshold selection
    #   20% FINAL TEST   -> touched ONCE at the very end
    #
    # Within development:
    #   80% of dev (64% total) -> cross-validation training folds
    #   20% of dev (16% total) -> calibration + threshold selection set
    #
    X_dev,  X_test,  y_dev,  y_test  = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=RANDOM_STATE
    )

    print(f"  Train (for CV):       {X_train.shape[0]} rows ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Calibration/Val:      {X_cal.shape[0]} rows ({X_cal.shape[0]/len(X)*100:.1f}%)")
    print(f"  Final Test (HOLDOUT): {X_test.shape[0]} rows ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test readmission rate: {y_test.mean()*100:.1f}%")
    print(f"  [!] Final test set is UNTOUCHED until Phase 12.")

    # ── Phase 4: Feature engineering comparison ────────────────────────────
    use_engineering, delta_fe_auc = compare_feature_sets(X_train, y_train)

    # ── Phase 6: Baseline comparison ──────────────────────────────────────
    # Phase 6 already runs 5-fold CV internally via run_baseline_comparison.
    # The XGBoost row there IS the default CV AUC — no separate run needed.
    comparison_df = phase_baseline_comparison(X_train, y_train)

    # Extract default XGBoost CV AUC from comparison table
    xgb_row      = comparison_df[comparison_df["Model"] == "XGBoost (default)"]
    default_cv_auc = float(xgb_row["Mean_ROC_AUC"].iloc[0])
    default_cv_std = float(xgb_row["Std_ROC_AUC"].iloc[0])
    print(f"\n  Default XGBoost CV ROC-AUC: {default_cv_auc:.4f} (+-{default_cv_std:.4f})")

    # ── Phase 7: Hyperparameter optimisation ──────────────────────────────
    best_pipeline, best_params = phase_hyperopt(X_train, y_train, default_cv_auc)

    # ── Phase 8: Early stopping ────────────────────────────────────────────
    recommended_n_est = phase_early_stopping(X_train, y_train, best_params)

    # Use early stopping recommendation if it differs meaningfully from search result
    tuned_n_est = best_params.get("n_estimators", 300)
    if abs(recommended_n_est - tuned_n_est) > 50:
        print(f"\n  Adjusting n_estimators: {tuned_n_est} -> {recommended_n_est} (early stopping)")
        best_params["n_estimators"] = recommended_n_est

    # ── Phase 9: Build and fit final pipeline ──────────────────────────────
    section("PHASE 9 — FINAL PIPELINE ASSEMBLY")
    print(f"  Feature engineering: {'ENABLED' if use_engineering else 'DISABLED (hurt CV AUC)'}")
    print(f"  Building pipeline with tuned params: {best_params}")

    final_pipeline = build_pipeline(
        xgb_params=best_params,
        use_feature_engineering=use_engineering,
    )
    # Fit on train portion (64% of total)
    final_pipeline.fit(X_train, y_train)
    print("  [OK] Pipeline fitted on training set")

    # ── Phase 10: Calibration ─────────────────────────────────────────────
    section("PHASE 10 — PROBABILITY CALIBRATION")

    y_proba_cal_raw = final_pipeline.predict_proba(X_cal)[:, 1]

    # Assess without calibration first
    cal_assess_raw = assess_calibration(
        y_cal, y_proba_cal_raw,
        output_dir=CHARTS_DIR,
        n_bins=10,
    )
    print(f"  Raw Brier score (on calibration set): {cal_assess_raw['brier_raw']:.4f}")

    # Apply Platt scaling on calibration set
    calibrated_model = apply_calibration(
        final_pipeline, X_cal, y_cal, method="sigmoid"
    )
    y_proba_cal_platt = calibrated_model.predict_proba(X_cal)[:, 1]

    # Reassess with calibration
    cal_assess_both = assess_calibration(
        y_cal, y_proba_cal_raw,
        y_proba_cal=y_proba_cal_platt,
        output_dir=CHARTS_DIR,
        n_bins=10,
    )
    print(f"  Calibrated Brier score:               {cal_assess_both.get('brier_calibrated', 'N/A'):.4f}")
    print(f"  Δ Brier (lower=better):               {cal_assess_both.get('delta_brier', 0):+.4f}")

    calibration_applied = bool(cal_assess_both.get("calibration_recommended", False))

    if calibration_applied:
        print("  -> Calibration APPLIED (Delta Brier > 0.005): using calibrated model")
        inference_model       = calibrated_model
        raw_pipeline_for_shap = final_pipeline   # PlattCalibratedPipeline.estimator
    else:
        print("  -> Calibration NOT applied (Delta Brier <= 0.005): raw scores sufficient")
        inference_model       = final_pipeline
        raw_pipeline_for_shap = final_pipeline

    # ── Phase 11: Threshold selection (on calibration set ONLY) ───────────
    section("PHASE 11 — THRESHOLD SELECTION (on calibration set)")
    print("  Method: Maximise F2 score (β=2, recall-favoured)")
    print("  IMPORTANT: Final test set NOT touched here.")

    y_proba_val_for_thr = inference_model.predict_proba(X_cal)[:, 1]
    best_threshold, val_metrics = select_threshold_on_val(
        y_cal, y_proba_val_for_thr, beta=2.0
    )

    print(f"\n  Selected threshold: {best_threshold:.4f}")
    print(f"  Validation F2:      {val_metrics['selection_fbeta']:.4f}")
    print(f"  Validation precision: {val_metrics['precision']:.4f}")
    print(f"  Validation recall:    {val_metrics['recall']:.4f}")
    print(f"  Validation F1:        {val_metrics['f1']:.4f}")
    print(f"  Flagged patients:     {val_metrics['flagged_pct']:.1f}%")

    # ── Phase 12: Final test evaluation (ONE SHOT) ─────────────────────────
    section("PHASE 12 — FINAL TEST SET EVALUATION (one-shot)")
    print("  [!] Test set is now unblinded for the first and ONLY time.\n")

    y_proba_test = inference_model.predict_proba(X_test)[:, 1]

    # Raw XGBoost on test (for fair comparison with baseline)
    y_proba_test_raw = final_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_at_threshold(
        y_test, y_proba_test, best_threshold, split_name="test"
    )

    print(f"  Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall:    {test_metrics['recall']:.4f}")
    print(f"  Test F1:        {test_metrics['f1']:.4f}")
    print(f"  Test F2:        {test_metrics['f2']:.4f}")
    print(f"  Flagged:        {test_metrics['flagged_pct']:.1f}% of test patients")
    print(f"\n  Confusion Matrix:")
    cm = test_metrics["confusion_matrix"]
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    # Calibration assessment on test set (reporting only)
    y_proba_test_cal = calibrated_model.predict_proba(X_test)[:, 1]
    test_cal = assess_calibration(
        y_test, y_proba_test_raw,
        y_proba_cal=y_proba_test_cal,
        output_dir=CHARTS_DIR,
    )
    print(f"\n  Test Brier (raw):        {test_cal['brier_raw']:.4f}")
    print(f"  Test Brier (calibrated): {test_cal.get('brier_calibrated', 'N/A'):.4f}")

    # Gains / lift table
    gains_table = compute_gains_table(y_test, y_proba_test)
    print("\n  Cumulative Gains & Lift:")
    print(gains_table[[
        "pct_patients", "readmissions_captured", "capture_rate_pct",
        "random_baseline_pct", "lift"
    ]].to_string(index=False))

    # Top-30% highlight
    row30 = gains_table[gains_table["pct_patients"] == 30].iloc[0]
    print(f"\n  [***] TOP 30%:")
    print(f"    Capture rate: {row30['capture_rate_pct']:.1f}%")
    print(f"    Lift:         {row30['lift']:.3f}x")
    print(f"    (Baseline result was: 40.3% capture, ~1.34x lift)")

    # Save charts
    plot_cumulative_gains(gains_table, CHARTS_DIR, model_label="XGBoost Final")
    plot_roc_curve(y_test, y_proba_test, CHARTS_DIR)

    # ── Phase 13: SHAP ────────────────────────────────────────────────────
    section("PHASE 13 — SHAP EXPLAINABILITY")
    try:
        # Use a sample of the test set for global importance (SHAP is slow on 5k rows)
        sample_size = min(500, len(X_test))
        X_shap_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)

        importance_df = compute_global_importance(
            raw_pipeline_for_shap, X_shap_sample,
            top_n=15, output_dir=CHARTS_DIR,
        )
        print(f"\n  Top 10 global features (mean |SHAP|):")
        print(importance_df.head(10).to_string(index=False))

        # Individual explanation — first test patient
        example_patient = X_test.iloc[[0]]
        explanation = explain_patient(raw_pipeline_for_shap, example_patient)
        print(f"\n  Example patient explanation:")
        print(f"    Raw risk score: {explanation['raw_risk_score']:.4f}")
        print(f"    Top risk-INCREASING features:")
        for f in explanation["top_increasing_risk"][:5]:
            print(f"      {f['feature']}: +{f['shap_value']:.4f}")
        print(f"    Top risk-DECREASING features:")
        for f in explanation["top_decreasing_risk"][:5]:
            print(f"      {f['feature']}: {f['shap_value']:.4f}")
        print(f"    Disclaimer: {explanation['disclaimer'][:80]}…")

        shap_success = True
    except Exception as e:
        print(f"  [!] SHAP computation failed: {e}")
        importance_df = pd.DataFrame()
        shap_success  = False

    # ── Phase 15: Save artifacts ───────────────────────────────────────────
    section("PHASE 15 — SAVING ARTIFACTS")

    # Save inference model (calibrated if applied, else raw)
    joblib.dump(inference_model, FINAL_PKL)
    print(f"  [OK] Model saved: {FINAL_PKL}")

    # Verify baseline is STILL intact
    assert BASELINE_PKL.exists(), "CRITICAL: Baseline model was deleted!"
    assert BASELINE_PKL.stat().st_size == baseline_size, \
        "CRITICAL: Baseline model was modified!"
    print(f"  [OK] Baseline model still intact: {BASELINE_PKL.name} ({baseline_size:,} bytes)")

    # Model metadata
    metadata = {
        "artifact": "readmission_model_final.pkl",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
            ],
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
            "method":       "RandomizedSearchCV",
            "n_iter":       40,
            "cv_folds":     5,
            "scoring":      "roc_auc",
            "random_state": RANDOM_STATE,
            "best_params":  best_params,
        },
        "early_stopping": {
            "recommended_n_estimators": recommended_n_est,
            "final_n_estimators":       best_params.get("n_estimators"),
        },
        "calibration": {
            "method":                 "sigmoid (Platt scaling)",
            "calibration_applied":    calibration_applied,
            "brier_raw_on_cal":       cal_assess_raw["brier_raw"],
            "brier_calibrated_on_cal": cal_assess_both.get("brier_calibrated"),
            "delta_brier":            cal_assess_both.get("delta_brier"),
            "brier_raw_on_test":      test_cal["brier_raw"],
            "brier_calibrated_on_test": test_cal.get("brier_calibrated"),
        },
        "threshold": {
            "value":              best_threshold,
            "selection_method":   "F2 maximisation on calibration set",
            "val_precision":      val_metrics["precision"],
            "val_recall":         val_metrics["recall"],
            "val_f1":             val_metrics["f1"],
            "val_f2":             val_metrics.get("selection_fbeta"),
            "val_flagged_pct":    val_metrics["flagged_pct"],
        },
        "final_test_metrics": {
            "roc_auc":        test_metrics["roc_auc"],
            "pr_auc":         test_metrics["pr_auc"],
            "precision":      test_metrics["precision"],
            "recall":         test_metrics["recall"],
            "f1":             test_metrics["f1"],
            "f2":             test_metrics["f2"],
            "flagged_pct":    test_metrics["flagged_pct"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "gains_table": gains_table.to_dict(orient="records"),
        "shap": {
            "implemented":  shap_success,
            "top_features": importance_df.head(10).to_dict(orient="records")
            if shap_success and not importance_df.empty else [],
        },
        "charts_dir": str(CHARTS_DIR),
        "baseline_model_preserved": str(BASELINE_PKL),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, default=str)
    print(f"  [OK] Metadata saved: {METADATA_PATH}")

    # ── Final summary ──────────────────────────────────────────────────────
    section("TRAINING COMPLETE — SUMMARY")
    print(f"  Final Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    print(f"  Final Test PR-AUC:   {test_metrics['pr_auc']:.4f}")
    print(f"  Operating threshold: {best_threshold:.4f}  (F2-selected on val set)")
    print(f"  Calibration:         {'Applied (Platt)' if calibration_applied else 'Not applied (Brier delta negligible)'}")
    print(f"  Top-30% capture:     {row30['capture_rate_pct']:.1f}%  (lift={row30['lift']:.3f}x)")
    print(f"  SHAP:                {'[OK] Implemented' if shap_success else '[!] Failed'}")
    print(f"\n  Artifacts:")
    print(f"    {FINAL_PKL}")
    print(f"    {METADATA_PATH}")
    print(f"    {CHARTS_DIR}/*.png")
    print(f"\n  Baseline preserved:  {BASELINE_PKL}  [OK]")


if __name__ == "__main__":
    main()
