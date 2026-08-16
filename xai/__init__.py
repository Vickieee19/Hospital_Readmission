"""XAI module for hospital readmission prediction explanations."""

from .explainer import explain_prediction, get_prediction_explanation

__all__ = ["explain_prediction", "get_prediction_explanation"]
