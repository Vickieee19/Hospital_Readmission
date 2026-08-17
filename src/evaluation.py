"""
evaluation.py
-------------
Full evaluation suite for the readmission-risk model.

Contents
--------
compute_metrics()            ROC-AUC, PR-AUC, confusion matrix, F1, P, R, MCC
select_threshold_by_mcc()    Choose threshold maximising MCC (balanced accuracy
                             on both classes equally) — PRIMARY method
select_threshold_on_val()    Legacy F-beta threshold selection (kept for reference)
evaluate_at_threshold()      Evaluate chosen threshold on any split
compute_gains_table()        Cumulative gains and lift at 10/20/.../100 %
plot_cumulative_gains()      Save cumulative gains + lift charts as PNG
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless rendering; no display required

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ── Core metrics ───────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Compute all relevant classification metrics.

    Parameters
    ----------
    y_true    : true binary labels
    y_proba   : model predicted probabilities (positive class)
    threshold : decision threshold for hard predictions;
                if None, only ranking metrics are returned

    Returns
    -------
    dict with keys: roc_auc, pr_auc, and (if threshold) precision,
                    recall, f1, f2, confusion_matrix, threshold,
                    flagged_count, flagged_pct
    """
    y_true = np.asarray(y_true)

    results: dict = {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc":  round(float(average_precision_score(y_true, y_proba)), 4),
    }

    if threshold is not None:
        y_pred = (y_proba >= threshold).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        results.update({
            "threshold":         threshold,
            "precision":         round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":            round(float(recall_score(y_true, y_pred, zero_division=0)),    4),
            "f1":                round(float(f1_score(y_true, y_pred, zero_division=0)),         4),
            "f2":                round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 4),
            "mcc":               round(float(matthews_corrcoef(y_true, y_pred)),                  4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)),            4),
            "confusion_matrix":  cm.tolist(),
            "flagged_count":     int(y_pred.sum()),
            "flagged_pct":       round(float(y_pred.mean() * 100), 2),
        })

    return results


# ── MCC threshold selection (PRIMARY — on validation data only) ──────────────

def select_threshold_by_mcc(
    y_val: np.ndarray | pd.Series,
    y_proba_val: np.ndarray,
) -> tuple[float, dict]:
    """
    Select the operating threshold that maximises the Matthews Correlation
    Coefficient (MCC) on validation data.

    Why MCC?
    --------
    MCC is the single most balanced metric for binary classification:
    - It accounts for ALL four cells of the confusion matrix (TP, TN, FP, FN)
    - A high MCC requires being correct on BOTH classes simultaneously
    - Unlike F2 (which only rewarded recall), MCC penalises false positives
      and false negatives equally
    - Range: -1 (worst) to +1 (perfect), 0 = random

    This is the PRIMARY threshold method replacing the F2-maximisation approach.

    Parameters
    ----------
    y_val       : true labels for validation split
    y_proba_val : predicted probabilities on validation split

    Returns
    -------
    (best_threshold, val_metrics_dict)
    """
    y_val = np.asarray(y_val)

    # Evaluate MCC at every unique predicted probability value
    # Use a grid of ~200 candidate thresholds for efficiency
    thresholds = np.unique(
        np.percentile(y_proba_val, np.linspace(5, 95, 200))
    )

    best_thr = 0.5
    best_mcc = -2.0

    for thr in thresholds:
        y_pred = (y_proba_val >= thr).astype(int)
        # Skip degenerate cases where all predictions are one class
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        mcc = matthews_corrcoef(y_val, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)

    val_metrics = compute_metrics(y_val, y_proba_val, threshold=best_thr)
    val_metrics["selection_metric"] = "MCC maximisation"
    val_metrics["selection_mcc"]    = round(best_mcc, 4)

    return best_thr, val_metrics


# ── Legacy F-beta threshold selection (kept for reference) ─────────────────

def select_threshold_on_val(
    y_val: np.ndarray | pd.Series,
    y_proba_val: np.ndarray,
    beta: float = 2.0,
    max_flagged_rate: float = 0.80,
) -> tuple[float, dict]:
    """
    Select the operating threshold on validation data.

    Strategy (two-stage)
    --------------------
    Stage 1 — F-beta maximisation with a flagged-rate cap:
        Maximise F-beta (beta=2 for recall-favoured) among thresholds
        that flag at most `max_flagged_rate` of the population.

        Rationale for the cap: with near-50/50 class balance and moderate
        AUC (~0.66), unconstrained F2 maximisation degenerates to flagging
        100% of patients (recall=1, precision=prevalence). This is
        mathematically correct but clinically useless for a risk-ranking
        model whose purpose is *prioritisation*.

    Stage 2 — Youden's J fallback:
        If no threshold within the cap produces F-beta > 0 (edge case),
        fall back to Youden's J statistic (TPR - FPR), which picks the
        threshold that maximally separates the two classes without
        weighting recall or precision.

    This function is called ONLY on validation/development data.
    The chosen threshold is applied ONCE to the final test set.

    Parameters
    ----------
    y_val           : true labels for validation split
    y_proba_val     : predicted probabilities on validation split
    beta            : F-beta parameter (default 2 for recall-focus)
    max_flagged_rate: maximum allowed fraction of population to flag
                      (default 0.80 — at most 80% flagged)

    Returns
    -------
    (best_threshold, val_metrics_dict)
    """
    y_val = np.asarray(y_val)
    n     = len(y_val)

    # ── Stage 1: constrained F-beta ───────────────────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba_val)

    # precision_recall_curve returns len(thresholds)+1 values for P and R
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    # Flagged rate at each threshold: fraction with proba >= threshold
    flagged_rates = np.array(
        [(y_proba_val >= thr).mean() for thr in thresholds]
    )

    # Only consider thresholds within the flagged-rate cap
    valid_mask = flagged_rates <= max_flagged_rate

    denom = (beta ** 2 * precisions + recalls)
    fbeta = np.where(denom > 0,
                     (1 + beta ** 2) * precisions * recalls / denom,
                     0.0)

    if valid_mask.any():
        # Best F-beta among valid (constrained) thresholds
        fbeta_constrained = np.where(valid_mask, fbeta, -np.inf)
        best_idx = int(np.argmax(fbeta_constrained))
        best_thr = float(thresholds[best_idx])
        selection_method = f"F{beta:.0f} (capped at {max_flagged_rate*100:.0f}% flagged)"
        best_fbeta = float(fbeta[best_idx])
    else:
        # ── Stage 2: Youden's J fallback ──────────────────────────────────
        from sklearn.metrics import roc_curve
        fpr, tpr, roc_thresholds = roc_curve(y_val, y_proba_val)
        youden_j = tpr - fpr
        best_roc_idx = int(np.argmax(youden_j))
        best_thr = float(roc_thresholds[best_roc_idx])
        selection_method = "Youden's J (fallback: no valid F-beta threshold)"
        best_fbeta = 0.0

    val_metrics = compute_metrics(y_val, y_proba_val, threshold=best_thr)
    val_metrics["selection_metric"] = selection_method
    val_metrics["selection_fbeta"]  = round(best_fbeta, 4)

    return best_thr, val_metrics



# ── Evaluate threshold on any split ───────────────────────────────────────

def evaluate_at_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    split_name: str = "test",
) -> dict:
    """
    Evaluate a pre-selected threshold on a data split.

    This is the ONLY time the threshold touches the final test set.
    """
    metrics = compute_metrics(y_true, y_proba, threshold=threshold)
    metrics["split"] = split_name
    return metrics


# ── Gains table ───────────────────────────────────────────────────────────

def compute_gains_table(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    percentiles: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute cumulative gains and lift at each percentile.

    Formula
    -------
    capture_rate = (readmissions in top-k%) / (total readmissions)
    lift         = capture_rate / (k/100)

    Parameters
    ----------
    y_true      : true binary labels
    y_proba     : predicted probabilities
    percentiles : list of integer percentages (default 10,20,...,100)

    Returns
    -------
    pd.DataFrame with columns:
        pct_patients, n_patients, readmissions_captured,
        total_readmissions, capture_rate_pct, random_baseline_pct, lift
    """
    if percentiles is None:
        percentiles = list(range(10, 110, 10))

    y_true  = np.asarray(y_true)
    results = pd.DataFrame({"proba": y_proba, "actual": y_true})
    results = results.sort_values("proba", ascending=False).reset_index(drop=True)

    total_readmissions = int(results["actual"].sum())
    n_total            = len(results)

    rows = []
    for pct in percentiles:
        n_patients   = max(1, int(n_total * pct / 100))
        top_df       = results.iloc[:n_patients]
        captured     = int(top_df["actual"].sum())
        capture_rate = captured / total_readmissions
        random_base  = pct / 100
        lift         = capture_rate / random_base if random_base > 0 else 0.0

        rows.append({
            "pct_patients":          pct,
            "n_patients":            n_patients,
            "readmissions_captured": captured,
            "total_readmissions":    total_readmissions,
            "capture_rate_pct":      round(capture_rate * 100, 2),
            "random_baseline_pct":   round(random_base  * 100, 2),
            "lift":                  round(lift, 3),
        })

    return pd.DataFrame(rows)


# ── Charts ────────────────────────────────────────────────────────────────

def plot_cumulative_gains(
    gains_table: pd.DataFrame,
    output_dir: Path,
    model_label: str = "XGBoost Final",
) -> None:
    """
    Save cumulative gains chart + lift chart as PNGs to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pct       = gains_table["pct_patients"].tolist()
    capture   = gains_table["capture_rate_pct"].tolist()
    lift_vals = gains_table["lift"].tolist()

    # ── Cumulative Gains ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0] + pct, [0] + capture, marker="o", linewidth=2, label=model_label)
    ax.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Random baseline")
    ax.set_xlabel("Percentage of Patients Prioritised (%)")
    ax.set_ylabel("Percentage of Readmissions Captured (%)")
    ax.set_title("Cumulative Gains Chart — Readmission Risk Model")
    ax.set_xticks(range(0, 110, 10))
    ax.set_yticks(range(0, 110, 10))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_gains.png", dpi=150)
    plt.close(fig)

    # ── Lift Chart ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pct, lift_vals, marker="o", linewidth=2, color="darkorange", label=model_label)
    ax.axhline(y=1.0, linestyle="--", color="gray", label="Random baseline (lift=1)")
    ax.set_xlabel("Percentage of Patients Prioritised (%)")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Chart — Readmission Risk Model")
    ax.set_xticks(range(0, 110, 10))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lift_chart.png", dpi=150)
    plt.close(fig)

    print(f"[Evaluation] Charts saved to {output_dir}")


def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    output_dir: Path,
    model_label: str = "XGBoost Final",
) -> None:
    """Save ROC curve PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val     = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"{model_label}  AUC={auc_val:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Readmission Risk Model")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
