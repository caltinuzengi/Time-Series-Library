"""
Evaluation metrics for forecasting and anomaly detection tasks.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# ---------------------------------------------------------------------------
# Forecasting metrics
# ---------------------------------------------------------------------------


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((pred - true) ** 2))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error.

    Args:
        pred: Predicted values.
        true: Ground-truth values.
        eps:  Small constant added to denominator to avoid division by zero.

    Returns:
        MAPE as a float (e.g. 0.05 means 5%).
    """
    return float(np.mean(np.abs((pred - true) / (np.abs(true) + eps))))


def evaluate_forecast(pred: np.ndarray, true: np.ndarray) -> dict:
    """Compute all forecasting metrics in one call.

    Args:
        pred: Predicted values, shape (N, ...).
        true: Ground-truth values, shape (N, ...).

    Returns:
        dict with keys: mse, mae, rmse, mape.
    """
    return {
        "mse": mse(pred, true),
        "mae": mae(pred, true),
        "rmse": rmse(pred, true),
        "mape": mape(pred, true),
    }


# ---------------------------------------------------------------------------
# Anomaly detection metrics
# ---------------------------------------------------------------------------


def precision_recall_f1(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 for binary anomaly labels.

    Args:
        pred_labels: Predicted binary labels (0/1), shape (N,).
        true_labels: Ground-truth binary labels (0/1), shape (N,).

    Returns:
        Tuple of (precision, recall, f1), each a float.
    """
    p = float(precision_score(true_labels, pred_labels, zero_division=0))
    r = float(recall_score(true_labels, pred_labels, zero_division=0))
    f = float(f1_score(true_labels, pred_labels, zero_division=0))
    return p, r, f


def auroc(scores: np.ndarray, true_labels: np.ndarray) -> float:
    """Area Under the ROC Curve.

    Args:
        scores:      Anomaly scores (higher = more anomalous), shape (N,).
        true_labels: Ground-truth binary labels (0/1), shape (N,).

    Returns:
        AUROC as a float.
    """
    return float(roc_auc_score(true_labels, scores))


def evaluate_anomaly(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict:
    """Compute all anomaly detection metrics in one call.

    Args:
        pred_labels: Predicted binary labels (0/1), shape (N,).
        true_labels: Ground-truth binary labels (0/1), shape (N,).
        scores:      Optional anomaly scores for AUROC computation.

    Returns:
        dict with keys: precision, recall, f1 (and auroc if scores provided).
    """
    p, r, f = precision_recall_f1(pred_labels, true_labels)
    result = {"precision": p, "recall": r, "f1": f}
    if scores is not None:
        result["auroc"] = auroc(scores, true_labels)
    return result
