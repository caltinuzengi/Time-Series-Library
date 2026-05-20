"""Forecasting experiment runner.

Supports any model registered in MODEL_REGISTRY.  New models are added by
inserting one line into the registry — no other code needs to change.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from data_provider.data_factory import get_dataloader
from exp.exp_base import ExpBase
from models.ModernTCN import ModernTCN
from models.PatchTST import PatchTST
from models.TimeMixer import TimeMixer
from models.TimesNet import TimesNet
from utils.metrics import evaluate_forecast
from utils.tools import EarlyStopping, load_checkpoint

# ---------------------------------------------------------------------------
# Model registry — add one line per new model (PatchTST, ModernTCN, …)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "TimesNet": TimesNet,
    "TimeMixer": TimeMixer,
    "PatchTST":  PatchTST,
    "ModernTCN": ModernTCN,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cuda_mem_str(device: torch.device) -> str:
    """Return a compact CUDA memory usage string, or '' on CPU."""
    if device.type != "cuda":
        return ""
    alloc = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    return f" | CUDA mem {alloc:.2f}/{reserved:.2f} GB"


# ---------------------------------------------------------------------------
# ExpForecasting
# ---------------------------------------------------------------------------

class ExpForecasting(ExpBase):
    """Supervised forecasting experiment.

    Args:
        args: Namespace with all hyper-parameters (see ``run.py``).
    """

    def __init__(self, args) -> None:
        super().__init__(args)
        # checkpoint_dir = the directory; checkpoint_path = the .pth file inside it
        self.checkpoint_dir  = Path(args.checkpoint_path)
        self.checkpoint_file = self.checkpoint_dir / "best_model.pth"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _build_model(self) -> nn.Module:
        model_cls = MODEL_REGISTRY.get(self.args.model)
        if model_cls is None:
            raise ValueError(
                f"Unknown model {self.args.model!r}. "
                f"Available: {list(MODEL_REGISTRY)}"
            )
        model = model_cls(self.args)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model built: {self.args.model} | params={n_params:,}")
        return model

    def _get_data(self, split: str):
        return get_dataloader(self.args, split)

    # ------------------------------------------------------------------
    def _log_device_info(self) -> None:
        if self.device.type == "cuda":
            idx = self.device.index or 0
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / 1e9
            logger.info(f"GPU : {props.name} | {total_gb:.1f} GB total")
        else:
            logger.info("Device: CPU")

    def _log_loader_info(self, loader, split: str) -> None:
        n_samples = len(loader.dataset)
        n_batches = len(loader)
        logger.info(
            f"{split.capitalize():5s} split: {n_samples:,} samples | "
            f"{n_batches} batches (batch_size={self.args.batch_size})"
        )

    # ------------------------------------------------------------------
    def _validate(self, val_loader) -> float:
        """Run one validation pass and return mean MSE loss."""
        self.model.eval()
        criterion = self._get_criterion()
        losses: list[float] = []

        with torch.inference_mode():
            for batch_x, batch_y, batch_x_mark, _ in val_loader:
                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                pred = self.model(batch_x, batch_x_mark)
                true = batch_y[:, -self.args.pred_len:, :].to(self.device)
                losses.append(criterion(pred, true).item())

        self.model.train()
        return float(np.mean(losses))

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Full training loop with early stopping and LR scheduling."""

        # ---- Data -------------------------------------------------------
        t_data = time.time()
        logger.info("Loading datasets …")
        train_loader = self._get_data("train")
        val_loader   = self._get_data("val")
        logger.info(f"Datasets loaded in {time.time() - t_data:.1f}s")

        self._log_device_info()
        self._log_loader_info(train_loader, "train")
        self._log_loader_info(val_loader,   "val")

        # ---- Optimizer / scheduler / stopper ----------------------------
        optimizer = self._get_optimizer()
        criterion = self._get_criterion()
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
        stopper   = EarlyStopping(patience=self.args.patience)

        logger.info(
            f"Training {self.args.model} on {self.args.data} | "
            f"pred_len={self.args.pred_len} | epochs={self.args.train_epochs} | "
            f"lr={self.args.learning_rate:.2e} | device={self.device}"
        )
        logger.info("-" * 60)

        # ---- Epoch loop -------------------------------------------------
        for epoch in range(1, self.args.train_epochs + 1):
            self.model.train()
            t_epoch = time.time()
            train_losses: list[float] = []
            t_data_total = 0.0
            t_fwd_total  = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch:03d}/{self.args.train_epochs} [train]",
                leave=False,
                unit="batch",
                dynamic_ncols=True,
            )
            t_batch_start = time.time()
            for batch_x, batch_y, batch_x_mark, _ in pbar:
                t_data_total += time.time() - t_batch_start

                optimizer.zero_grad()

                t_fwd_start = time.time()
                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                pred = self.model(batch_x, batch_x_mark)
                true = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss = criterion(pred, true)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
                t_fwd_total += time.time() - t_fwd_start

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                t_batch_start = time.time()

            pbar.close()

            # Validation
            t_val = time.time()
            train_loss = float(np.mean(train_losses))
            val_loss   = self._validate(val_loader)
            t_val_elapsed = time.time() - t_val
            elapsed = time.time() - t_epoch
            lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:03d}/{self.args.train_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s "
                f"[data={t_data_total:.1f}s fwd/bwd={t_fwd_total:.1f}s val={t_val_elapsed:.1f}s]"
                + _cuda_mem_str(self.device)
            )

            self.epoch_logs.append({
                "epoch":      epoch,
                "train_loss": round(train_loss, 6),
                "val_loss":   round(val_loss,   6),
                "lr":         lr,
                "elapsed_s":  round(elapsed, 2),
            })

            scheduler.step(val_loss)
            stopper(val_loss, self.model, str(self.checkpoint_dir))  # dir, not .pth file

            if stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch} (patience={self.args.patience}).")
                break

        logger.info("-" * 60)
        load_checkpoint(self.model, str(self.checkpoint_dir))  # dir → appends best_model.pth inside
        logger.info(f"Best model restored from {self.checkpoint_file}")

    # ------------------------------------------------------------------
    def test(self) -> dict:
        """Evaluate on the test set.

        Returns:
            Dict with keys ``mse``, ``mae``, ``rmse``, ``mape``.
        """
        logger.info("Starting test evaluation …")
        load_checkpoint(self.model, str(self.checkpoint_dir))
        test_loader = self._get_data("test")
        self._log_loader_info(test_loader, "test")

        self.model.eval()
        preds: list[np.ndarray] = []
        trues: list[np.ndarray] = []

        t_test = time.time()
        with torch.inference_mode():
            for batch_x, batch_y, batch_x_mark, _ in tqdm(
                test_loader, desc="Testing", leave=False, unit="batch"
            ):
                batch_x      = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                pred = self.model(batch_x, batch_x_mark)
                true = batch_y[:, -self.args.pred_len:, :]

                preds.append(pred.cpu().numpy())
                trues.append(true.numpy())

        preds_np = np.concatenate(preds, axis=0)
        trues_np = np.concatenate(trues, axis=0)

        metrics = evaluate_forecast(preds_np, trues_np)
        logger.info(
            f"Test done in {time.time() - t_test:.1f}s | "
            f"MSE={metrics['mse']:.4f} | MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | MAPE={metrics['mape']:.4f}"
        )
        return metrics
