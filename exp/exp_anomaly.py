"""Anomaly Detection experiment runner.

Training:  minimize MSE reconstruction loss (unsupervised, no labels needed).
Testing:   combined train+test scores → percentile threshold → point-adjust F1.

This module is fully independent of ``exp_forecasting.py``.
The existing forecasting pipeline is not affected.

Reference approach:
    Wu et al., "TimesNet", ICLR 2023 — Section 5.3 / Appendix B.
    Combined-distribution threshold: percentile(concat(train_scores, test_scores),
    100 - anomaly_ratio).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm.auto import tqdm

from data_provider.data_factory import get_dataloader
from exp.exp_base import ExpBase
from models.ModernTCN import ModernTCN
from models.PatchTST import PatchTST
from models.TimeMixer import TimeMixer
from models.TimesNet import TimesNet
from utils.anomaly_metrics import compute_anomaly_metrics
from utils.tools import EarlyStopping, load_checkpoint, save_checkpoint

MODEL_REGISTRY_AD: dict[str, type[nn.Module]] = {
    "TimesNet": TimesNet,
    "TimeMixer": TimeMixer,
    "PatchTST":  PatchTST,
    "ModernTCN": ModernTCN,
}


class ExpAnomaly(ExpBase):
    """Reconstruction-based unsupervised anomaly detection experiment."""

    def _build_model(self) -> nn.Module:
        if self.args.model not in MODEL_REGISTRY_AD:
            raise ValueError(
                f"Model {self.args.model!r} not supported for anomaly detection. "
                f"Available: {list(MODEL_REGISTRY_AD)}"
            )
        model = MODEL_REGISTRY_AD[self.args.model](self.args)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model built: {self.args.model} | params={n_params:,}")
        return model

    def _get_data(self, split: str):
        return get_dataloader(self.args, split)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Train with MSE reconstruction loss, save best-train-loss checkpoint."""
        train_loader = self._get_data("train")
        criterion    = nn.MSELoss()
        optimizer    = self._get_optimizer()
        early_stop   = EarlyStopping(patience=self.args.patience, delta=1e-4)

        logger.info(
            f"Training {self.args.model} on {self.args.data} (anomaly_detection) | "
            f"epochs={self.args.train_epochs} | lr={self.args.learning_rate:.2e} | "
            f"device={self.device}"
        )
        logger.info("-" * 60)

        for epoch in range(1, self.args.train_epochs + 1):
            t0 = time.time()
            self.model.train()
            losses: list[float] = []

            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch:03d}",
                leave=False,
                dynamic_ncols=True,
            ):
                # AnomalyDataset train split returns a plain Tensor
                x = batch.float().to(self.device)           # (B, T, C)

                recon = self.model.anomaly_detection(x)
                loss  = criterion(recon, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            train_loss = float(np.mean(losses))
            elapsed    = time.time() - t0

            logger.info(
                f"Epoch {epoch:03d}/{self.args.train_epochs} | "
                f"train_loss={train_loss:.4f} | {elapsed:.1f}s"
            )

            self.epoch_logs.append({
                "epoch":      epoch,
                "train_loss": round(train_loss, 6),
                "elapsed_s":  round(elapsed, 2),
            })

            early_stop(train_loss, self.model, self.args.checkpoint_path)
            if early_stop.early_stop:
                logger.info("Early stopping triggered.")
                break

        logger.info("-" * 60)
        logger.info(f"Loading best checkpoint from {self.args.checkpoint_path}")
        load_checkpoint(self.model, self.args.checkpoint_path, map_location=self.device)

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_raw_scores(
        self, loader
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute per-timestep reconstruction MSE (mean over channels).

        Args:
            loader: DataLoader whose batches are either:
                - plain ``Tensor (B, T, C)``  (train split), or
                - ``(Tensor(B, T, C), Tensor(B, T))`` tuple (test split).

        Returns:
            ``(scores, labels)`` both flattened to 1-D.
            ``labels`` is ``None`` for the training split.
        """
        self.model.eval()
        point_criterion = nn.MSELoss(reduction="none")
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x, labels = batch
                    all_labels.append(labels.numpy())
                else:
                    x = batch

                x     = x.float().to(self.device)
                recon = self.model.anomaly_detection(x)

                # Mean over channel dim → (B, T) per-timestep reconstruction error
                score = point_criterion(recon, x).mean(dim=-1)  # (B, T)
                all_scores.append(score.cpu().numpy())

        scores = np.concatenate(all_scores).reshape(-1)
        labels = np.concatenate(all_labels).reshape(-1) if all_labels else None
        return scores, labels

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def test(self) -> dict:
        """Evaluate using combined-distribution threshold and point-adjust F1.

        Steps:
        1. Compute reconstruction scores on training data (for threshold calibration).
        2. Compute reconstruction scores + labels on test data.
        3. Threshold = percentile(concat(train, test), 100 - anomaly_ratio).
        4. Compute point-wise and point-adjusted precision / recall / F1 / AUROC.

        Returns:
            Metrics dictionary (see :func:`~utils.anomaly_metrics.compute_anomaly_metrics`).
        """
        logger.info("Starting anomaly detection evaluation …")
        load_checkpoint(self.model, self.args.checkpoint_path, map_location=self.device)

        train_loader = self._get_data("train")
        test_loader  = self._get_data("test")

        # Step 1: reconstruction scores on normal (train) data
        logger.info("Computing train scores …")
        train_scores, _ = self._compute_raw_scores(train_loader)

        # Step 2: reconstruction scores + ground-truth labels on test data
        logger.info("Computing test scores …")
        test_scores, test_labels = self._compute_raw_scores(test_loader)

        if test_labels is None:
            raise RuntimeError(
                "Test loader returned no labels. "
                "Ensure the test split of AnomalyDataset returns (x, label) tuples."
            )

        # Step 3: combined-distribution threshold
        combined  = np.concatenate([train_scores, test_scores])
        threshold = np.percentile(combined, 100.0 - self.args.anomaly_ratio)
        logger.info(
            f"Threshold = {threshold:.6f}  "
            f"(anomaly_ratio = {self.args.anomaly_ratio:.1f}%)"
        )

        # Step 4: metrics
        gt      = test_labels.astype(int)
        metrics = compute_anomaly_metrics(test_scores, gt, threshold)

        logger.info(
            f"F1={metrics['f1']:.4f} | "
            f"PA-F1={metrics['f1_pa']:.4f} | "
            f"Precision_PA={metrics['precision_pa']:.4f} | "
            f"Recall_PA={metrics['recall_pa']:.4f} | "
            f"AUROC={metrics['auroc']:.4f}"
        )
        return metrics
