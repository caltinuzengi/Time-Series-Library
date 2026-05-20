"""Smoke-test for anomaly_detection() on all 4 models.

Creates synthetic data in a temporary directory that mirrors the SMD
directory layout — no real SMD files required.

What is tested:
  1. All 4 models can be instantiated with SMD-like configs.
  2. anomaly_detection() returns a tensor of the correct shape (B, seq_len, C).
  3. All output values are finite.
  4. utils.anomaly_metrics works end-to-end (compute + point_adjust).
  5. AnomalyDataset can load the mock files via the data pipeline.

Usage:
  uv run check-anomaly
"""

from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch


# ---------------------------------------------------------------------------
# SMD-like config (38 variates, seq_len=100)
# ---------------------------------------------------------------------------

def _smd_configs(seq_len: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        # core sizes
        seq_len            = seq_len,
        pred_len           = 96,          # needed by TimesNet/PatchTST heads (not used in recon)
        label_len          = 48,
        enc_in             = 38,
        c_out              = 38,
        # shared arch
        d_model            = 32,
        d_ff               = 64,
        e_layers           = 2,
        dropout            = 0.0,
        # TimesNet-specific
        top_k              = 3,
        num_kernels        = 4,
        # PatchTST-specific
        n_heads            = 4,
        patch_len          = 16,
        stride             = 8,
        # ModernTCN-specific
        patch_size         = 8,
        patch_stride       = 8,
        large_kernel       = 25,
        small_kernel       = 5,
        ffn_ratio          = 2,
        # TimeMixer-specific
        down_sampling_layers = 2,
        down_sampling_window = 2,
        moving_avg           = 7,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_model(name: str, cfg: SimpleNamespace, x: torch.Tensor) -> None:
    """Instantiate model ``name``, call anomaly_detection, verify shape."""
    from models.ModernTCN import ModernTCN
    from models.PatchTST import PatchTST
    from models.TimeMixer import TimeMixer
    from models.TimesNet import TimesNet

    registry = {
        "TimesNet": TimesNet,
        "TimeMixer": TimeMixer,
        "PatchTST":  PatchTST,
        "ModernTCN": ModernTCN,
    }

    model = registry[name](cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name:12s}  params={n_params:>9,}", end="  ")

    with torch.no_grad():
        out = model.anomaly_detection(x)

    B, T, C = x.shape
    assert out.shape == (B, T, C), (
        f"{name}: expected output shape {(B, T, C)}, got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), f"{name}: non-finite values in output"

    print(f"output={tuple(out.shape)}  ✓")


def _check_metrics() -> None:
    """Quick unit-test for compute_anomaly_metrics + point_adjust."""
    from utils.anomaly_metrics import compute_anomaly_metrics, point_adjust

    rng    = np.random.default_rng(0)
    T      = 500
    gt     = (rng.random(T) > 0.9).astype(int)
    scores = rng.random(T).astype(np.float32)
    scores[gt == 1] += 0.3                         # anomalies have slightly higher scores

    threshold = np.percentile(scores, 95)
    pred      = (scores > threshold).astype(int)
    pred_pa   = point_adjust(pred, gt)

    # sanity: point-adjusted recall ≥ raw recall (PA can only add detections)
    from sklearn.metrics import recall_score
    raw_recall = recall_score(gt, pred, zero_division=0)
    pa_recall  = recall_score(gt, pred_pa, zero_division=0)
    assert pa_recall >= raw_recall, "PA recall should be ≥ raw recall"

    metrics = compute_anomaly_metrics(scores, gt, threshold)
    required_keys = {"f1", "precision", "recall", "f1_pa", "precision_pa",
                     "recall_pa", "auroc", "threshold"}
    assert required_keys == set(metrics.keys()), f"Missing keys: {required_keys - set(metrics.keys())}"

    print(f"  anomaly_metrics  F1={metrics['f1']:.3f}  PA-F1={metrics['f1_pa']:.3f}  AUROC={metrics['auroc']:.3f}  ✓")


def _check_data_pipeline(tmp_dir: str, seq_len: int = 100) -> None:
    """Verify AnomalyDataset can load mock SMD-formatted files."""
    from data_provider.data_factory import get_dataloader

    # Fake anomaly_ratio not needed here; just test that loader works.
    cfg = SimpleNamespace(
        task            = "anomaly_detection",
        data            = "SMD",
        root_path       = tmp_dir,
        seq_len         = seq_len,
        batch_size      = 8,
        num_workers     = 0,
        anomaly_ratio   = 1.0,
    )

    train_loader = get_dataloader(cfg, "train")
    test_loader  = get_dataloader(cfg, "test")

    # Train: should return plain tensors
    for batch in train_loader:
        assert not isinstance(batch, (list, tuple)), "Train batch should be a plain Tensor"
        assert batch.shape[-1] == 38, f"Expected 38 features, got {batch.shape[-1]}"
        break

    # Test: should return (x, label) tuples
    for batch in test_loader:
        assert isinstance(batch, (list, tuple)), "Test batch should be (x, label) tuple"
        x, label = batch
        assert x.shape[-1] == 38
        assert label.ndim == 2, "labels should be 2-D: (B, T)"
        break

    print(f"  data_pipeline    shapes OK  ✓")


# ---------------------------------------------------------------------------
# Mock SMD data creation
# ---------------------------------------------------------------------------

def _make_mock_smd(tmp_dir: str, n_features: int = 38, n_train: int = 3000,
                   n_test: int = 500) -> None:
    """Create minimal SMD-like directory structure with random float data."""
    rng = np.random.default_rng(42)

    for sub in ("train", "test", "test_label"):
        os.makedirs(os.path.join(tmp_dir, "SMD", sub), exist_ok=True)

    # One machine file is enough for the smoke-test
    machine = "machine-1-1"

    # train
    data_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    np.savetxt(
        os.path.join(tmp_dir, "SMD", "train", f"{machine}.txt"),
        data_train, delimiter=",", fmt="%.6f",
    )

    # test
    data_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
    np.savetxt(
        os.path.join(tmp_dir, "SMD", "test", f"{machine}.txt"),
        data_test, delimiter=",", fmt="%.6f",
    )

    # test_label  (sparse anomalies)
    labels = np.zeros(n_test, dtype=np.int32)
    labels[200:220] = 1
    labels[350:360] = 1
    np.savetxt(
        os.path.join(tmp_dir, "SMD", "test_label", f"{machine}.txt"),
        labels, delimiter=",", fmt="%d",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("check-anomaly — anomaly detection smoke-test")
    print("=" * 60)

    cfg = _smd_configs()
    B   = 4
    x   = torch.randn(B, cfg.seq_len, cfg.enc_in)

    # ---- 1. Model outputs ----
    print("\n[1] Model anomaly_detection() output shapes:")
    for name in ["TimesNet", "TimeMixer", "PatchTST", "ModernTCN"]:
        _check_model(name, cfg, x)

    # ---- 2. Metrics ----
    print("\n[2] Anomaly metrics (point_adjust + compute_anomaly_metrics):")
    _check_metrics()

    # ---- 3. Data pipeline ----
    print("\n[3] Data pipeline (AnomalyDataset with mock SMD files):")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_mock_smd(tmp_dir)
        _check_data_pipeline(tmp_dir, seq_len=cfg.seq_len)

    print("\n" + "=" * 60)
    print("All checks PASSED ✓")
    print("=" * 60)
