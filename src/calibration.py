"""
calibration.py
--------------
Probability calibration assessment and optional calibration for the
readmission-risk pipeline.

Why calibration matters
-----------------------
XGBoost outputs raw "scores" that are not guaranteed to represent true
probabilities. Showing "73% chance of readmission" based on raw XGBoost
output is scientifically misleading unless calibration has been verified.

This module:
1. Assesses calibration quality on validation data (Brier score + reliability diagram)
2. Optionally wraps the fitted estimator in CalibratedClassifierCV if
   calibration materially improves reliability (Δ Brier > 0.005)
3. Returns a clear flag indicating whether calibration was applied

Design decisions
----------------
- Calibration is fitted on a held-out CALIBRATION set (a subset of the
  dev/train portion), not on the final test set.
- Sigmoid (Platt) scaling is used; isotonic regression requires more
  samples for stable results and can overfit on smaller calibration sets.
- The calibration threshold for application is Δ Brier > 0.005
  (a practically meaningful improvement, not just numerical noise).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline


# ── Assessment ─────────────────────────────────────────────────────────────

def assess_calibration(
    y_true: np.ndarray | pd.Series,
    y_proba_raw: np.ndarray,
    y_proba_cal: np.ndarray | None = None,
    output_dir: Path | None = None,
    n_bins: int = 10,
) -> dict:
    """
    Evaluate calibration quality with Brier score and reliability diagram.

    Parameters
    ----------
    y_true       : true binary labels (validation or test)
    y_proba_raw  : raw model probabilities before calibration
    y_proba_cal  : calibrated probabilities (optional; for comparison)
    output_dir   : if provided, save reliability diagram PNG here
    n_bins       : number of bins for calibration curve

    Returns
    -------
    dict with keys:
        brier_raw, brier_calibrated (if provided), delta_brier,
        calibration_recommended
    """
    y_true = np.asarray(y_true)

    brier_raw = float(brier_score_loss(y_true, y_proba_raw))
    result    = {"brier_raw": round(brier_raw, 4)}

    if y_proba_cal is not None:
        brier_cal   = float(brier_score_loss(y_true, y_proba_cal))
        delta       = brier_raw - brier_cal
        result["brier_calibrated"]       = round(brier_cal, 4)
        result["delta_brier"]            = round(delta, 4)
        result["calibration_recommended"] = delta > 0.005
    else:
        result["calibration_recommended"] = None

    # ── Reliability diagram ──────────────────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 6))

        frac_pos_raw, mean_pred_raw = calibration_curve(
            y_true, y_proba_raw, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_pred_raw, frac_pos_raw,
                marker="o", linewidth=2, label="Raw XGBoost")

        if y_proba_cal is not None:
            frac_pos_cal, mean_pred_cal = calibration_curve(
                y_true, y_proba_cal, n_bins=n_bins, strategy="uniform"
            )
            ax.plot(mean_pred_cal, frac_pos_cal,
                    marker="s", linewidth=2, linestyle="--",
                    label="Calibrated (Platt)")

        ax.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Reliability Diagram — Readmission Risk Model")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "calibration_curve.png", dpi=150)
        plt.close(fig)
        print(f"[Calibration] Reliability diagram saved to {output_dir}")

    return result


# ── Manual Platt calibration wrapper ──────────────────────────────────────

class PlattCalibratedPipeline:
    """
    Wraps a fitted sklearn Pipeline with Platt (sigmoid) calibration.

    Why not CalibratedClassifierCV(cv="prefit")?
    ─────────────────────────────────────────────
    sklearn 1.9.0 removed the cv="prefit" option. This class implements
    the same Platt scaling (sigmoid calibration) manually:

        1. Get raw probabilities from the fitted base pipeline
        2. Fit a LogisticRegression on those probabilities (1D input)
           → this IS Platt scaling
        3. At inference, chain both steps

    The result is serializable with joblib and has the same predict_proba()
    interface as any sklearn estimator, so it is a drop-in replacement.

    Attributes
    ----------
    base_pipeline   : the original fitted sklearn Pipeline
    calibrator      : fitted LogisticRegression on raw probabilities
    calibration_set_size : number of samples used to fit calibration
    """

    def __init__(self, base_pipeline):
        self.base_pipeline        = base_pipeline
        self.calibrator           = None
        self.calibration_set_size = 0

    def fit(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> "PlattCalibratedPipeline":
        """
        Fit Platt scaling on the calibration set.

        Parameters
        ----------
        X_cal : raw patient DataFrame (calibration split)
        y_cal : true binary labels
        """
        from sklearn.linear_model import LogisticRegression

        # Raw probabilities from the already-fitted base pipeline
        raw_proba = self.base_pipeline.predict_proba(X_cal)[:, 1]

        # Fit logistic regression on raw proba -> true labels
        # This is mathematically identical to Platt scaling
        self.calibrator = LogisticRegression(C=1.0, max_iter=1000)
        self.calibrator.fit(raw_proba.reshape(-1, 1), y_cal)

        self.calibration_set_size = len(y_cal)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Return calibrated probabilities.

        Returns array of shape (n_samples, 2) like standard sklearn.
        """
        if self.calibrator is None:
            raise RuntimeError("Call .fit() before .predict_proba()")

        raw_proba = self.base_pipeline.predict_proba(X)[:, 1]
        cal_proba = self.calibrator.predict_proba(raw_proba.reshape(-1, 1))
        return cal_proba   # shape (n, 2): [prob_neg, prob_pos]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Return hard binary predictions at given threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def estimator(self):
        """Expose base pipeline for SHAP extraction."""
        return self.base_pipeline

    def __repr__(self) -> str:
        return (
            f"PlattCalibratedPipeline("
            f"calibration_set_size={self.calibration_set_size})"
        )


def apply_calibration(
    fitted_pipeline: Pipeline,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    method: str = "sigmoid",
) -> "PlattCalibratedPipeline":
    """
    Apply post-hoc Platt (sigmoid) calibration to a fitted pipeline.

    [!]  fitted_pipeline must already be fitted.
    [!]  X_cal / y_cal must be a HELD-OUT calibration set,
         never the final test set.

    Parameters
    ----------
    fitted_pipeline : fitted sklearn Pipeline
    X_cal           : calibration features (raw DataFrame)
    y_cal           : calibration labels
    method          : currently only "sigmoid" is implemented
                      (isotonic would need ~5k+ samples for stability)

    Returns
    -------
    PlattCalibratedPipeline  (has .predict_proba(), .predict(), .estimator)
    """
    calibrated = PlattCalibratedPipeline(base_pipeline=fitted_pipeline)
    calibrated.fit(X_cal, y_cal)
    print(f"  [Calibration] Platt scaling fitted on {len(y_cal)} calibration samples.")
    return calibrated
