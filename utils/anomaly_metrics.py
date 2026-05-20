"""Anomaly detection evaluation metrics.

Standard functions for time-series anomaly detection benchmarks:

    point_adjust          — detection-adjustment protocol (segment-level)
    compute_anomaly_metrics — precision / recall / F1 (raw & point-adjusted) + AUROC

Reference:
    Xu et al., "Anomaly Transformer", ICLR 2022.
    Protocol also used in TimesNet (Wu et al., ICLR 2023) Table 6.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def point_adjust(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Standard point-adjustment (PA) protocol.

    For each contiguous anomaly *segment* in ``gt``:
    if ANY point within the segment is predicted anomalous (``pred == 1``),
    all points in that segment are set to 1 in the adjusted prediction.

    This accounts for detectors that fire slightly before/after the labelled
    segment boundary, and is the primary metric in the literature.

    Args:
        pred: Binary prediction array, shape ``(T,)``.
        gt:   Ground-truth binary label array, shape ``(T,)``.

    Returns:
        Adjusted binary prediction array, same shape as ``pred``.
    """
    pred_adj = pred.copy()
    n = len(gt)
    i = 0
    while i < n:
        if gt[i] == 1:
            # Find end of this contiguous anomaly segment [i, j)
            j = i
            while j < n and gt[j] == 1:
                j += 1
            # If ANY point in [i, j) was flagged, flag the whole segment
            if np.any(pred[i:j] == 1):
                pred_adj[i:j] = 1
            i = j
        else:
            i += 1
    return pred_adj


def compute_anomaly_metrics(
    scores: np.ndarray,
    gt: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute standard anomaly detection metrics.

    Args:
        scores:    Continuous anomaly score per timestep, shape ``(T,)``.
        gt:        Binary ground-truth labels, shape ``(T,)``.
        threshold: Decision boundary — points with ``score > threshold`` are
                   predicted anomalous.

    Returns:
        Dictionary with the following keys:

        ============  ====================================================
        ``f1``        Point-wise F1 score
        ``precision`` Point-wise precision
        ``recall``    Point-wise recall
        ``f1_pa``     **Point-adjusted F1** (primary benchmark metric)
        ``precision_pa`` Point-adjusted precision
        ``recall_pa`` Point-adjusted recall
        ``auroc``     Area under the ROC curve (NaN if ``gt`` is constant)
        ``threshold`` The threshold that was applied
        ============  ====================================================
    """
    pred    = (scores > threshold).astype(int)
    pred_pa = point_adjust(pred, gt)

    try:
        auroc = float(roc_auc_score(gt, scores))
    except ValueError:
        auroc = float("nan")

    return {
        "f1":           float(f1_score(gt, pred,    zero_division=0)),
        "precision":    float(precision_score(gt, pred,    zero_division=0)),
        "recall":       float(recall_score(gt, pred,    zero_division=0)),
        "f1_pa":        float(f1_score(gt, pred_pa, zero_division=0)),
        "precision_pa": float(precision_score(gt, pred_pa, zero_division=0)),
        "recall_pa":    float(recall_score(gt, pred_pa, zero_division=0)),
        "auroc":        auroc,
        "threshold":    float(threshold),
    }
