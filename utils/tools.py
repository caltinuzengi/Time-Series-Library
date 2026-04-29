"""
Training utilities: early stopping, checkpoint save/load.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        delta:    Minimum improvement threshold. An update is counted only
                  when val_loss improves by more than ``delta``.
    """

    def __init__(self, patience: int = 3, delta: float = 0.0) -> None:
        self.patience = patience
        self.delta = delta
        self.counter: int = 0
        self.best_score: float | None = None
        self.early_stop: bool = False
        self.val_loss_min: float = np.inf

    def __call__(self, val_loss: float, model: nn.Module, path: str | Path) -> None:
        """Check improvement and conditionally save the model.

        Args:
            val_loss: Current epoch validation loss.
            model:    PyTorch model to checkpoint.
            path:     Directory where ``best_model.pth`` will be saved.
        """
        score = -val_loss  # higher is better

        if self.best_score is None:
            self.best_score = score
            save_checkpoint(model, path)
            logger.debug(
                f"EarlyStopping: initial checkpoint (val_loss={val_loss:.6f})"
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs "
                f"(val_loss={val_loss:.6f}, best={-self.best_score:.6f})"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save_checkpoint(model, path)
            self.counter = 0
            logger.debug(
                f"EarlyStopping: improved (val_loss={val_loss:.6f}) — checkpoint saved"
            )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    """Save model state dict to ``{path}/best_model.pth``.

    Args:
        model: PyTorch model.
        path:  Target directory (will be created if absent).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = path / "best_model.pth"
    torch.save(model.state_dict(), checkpoint_file)
    logger.debug(f"Checkpoint saved → {checkpoint_file}")


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> nn.Module:
    """Load state dict from ``{path}/best_model.pth`` into *model* in-place.

    Args:
        model:        PyTorch model (must match the saved architecture).
        path:         Directory containing ``best_model.pth``.
        map_location: Passed to ``torch.load`` for device remapping.

    Returns:
        The same model with loaded weights (for chaining convenience).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    checkpoint_file = Path(path) / "best_model.pth"
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_file}\n"
            "Run training first to generate a checkpoint."
        )
    state = torch.load(checkpoint_file, map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    logger.debug(f"Checkpoint loaded ← {checkpoint_file}")
    return model
