"""Forecasting experiment runner.

Supports any model registered in MODEL_REGISTRY.  New models are added by
inserting one line into the registry — no other code needs to change.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_provider.data_factory import get_dataloader
from exp.exp_base import ExpBase
from models.TimesNet import TimesNet
from utils.metrics import evaluate_forecast
from utils.tools import EarlyStopping, load_checkpoint, save_checkpoint

# ---------------------------------------------------------------------------
# Model registry — add one line per new model (PatchTST, ModernTCN, …)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "TimesNet": TimesNet,
    # "TimeMixer": TimeMixer,   # Faz 4
    # "PatchTST":  PatchTST,   # Faz 5
    # "ModernTCN": ModernTCN,  # Faz 6
}


class ExpForecasting(ExpBase):
    """Supervised forecasting experiment.

    Args:
        args: Namespace with all hyper-parameters (see ``run.py``).
    """

    def __init__(self, args) -> None:
        super().__init__(args)
        self.checkpoint_path = Path(args.checkpoint_path) / "best_model.pth"
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _build_model(self) -> nn.Module:
        model_cls = MODEL_REGISTRY.get(self.args.model)
        if model_cls is None:
            raise ValueError(
                f"Unknown model {self.args.model!r}. "
                f"Available: {list(MODEL_REGISTRY)}"
            )
        return model_cls(self.args)

    def _get_data(self, split: str):
        return get_dataloader(self.args, split)

    # ------------------------------------------------------------------
    def _validate(self, val_loader) -> float:
        """Run one validation pass and return mean MSE loss."""
        self.model.eval()
        criterion = self._get_criterion()
        losses: list[float] = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                pred = self.model(batch_x, batch_x_mark)          # (B, pred_len, C)
                true = batch_y[:, -self.args.pred_len:, :].to(self.device)
                losses.append(criterion(pred, true).item())

        self.model.train()
        return float(np.mean(losses))

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Full training loop with early stopping and LR scheduling."""
        train_loader = self._get_data("train")
        val_loader = self._get_data("val")

        optimizer = self._get_optimizer()
        criterion = self._get_criterion()
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5, verbose=False
        )
        stopper = EarlyStopping(patience=self.args.patience)

        logger.info(
            f"Training {self.args.model} on {self.args.data} | "
            f"pred_len={self.args.pred_len} | device={self.device}"
        )

        for epoch in range(1, self.args.train_epochs + 1):
            self.model.train()
            t0 = time.time()
            train_losses: list[float] = []

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                pred = self.model(batch_x, batch_x_mark)           # (B, pred_len, C)
                true = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss = criterion(pred, true)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))
            val_loss = self._validate(val_loader)
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:03d}/{self.args.train_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            scheduler.step(val_loss)
            stopper(val_loss, self.model, str(self.checkpoint_path))

            if stopper.early_stop:
                logger.info("Early stopping triggered.")
                break

        # Restore best weights
        load_checkpoint(self.model, str(self.checkpoint_path))
        logger.info(f"Best model loaded from {self.checkpoint_path}")

    # ------------------------------------------------------------------
    def test(self) -> dict:
        """Evaluate on the test set.

        Returns:
            Dict with keys ``mse``, ``mae``, ``rmse``, ``mape``.
        """
        load_checkpoint(self.model, str(self.checkpoint_path))
        test_loader = self._get_data("test")

        self.model.eval()
        preds: list[np.ndarray] = []
        trues: list[np.ndarray] = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, _ in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                pred = self.model(batch_x, batch_x_mark)           # (B, pred_len, C)
                true = batch_y[:, -self.args.pred_len:, :]         # (B, pred_len, C)

                preds.append(pred.cpu().numpy())
                trues.append(true.numpy())

        preds_np = np.concatenate(preds, axis=0)
        trues_np = np.concatenate(trues, axis=0)

        metrics = evaluate_forecast(preds_np, trues_np)
        logger.info(
            f"Test results — MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f}"
        )
        return metrics
