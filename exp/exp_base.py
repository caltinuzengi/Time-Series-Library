"""Abstract base class for all experiment runners."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class ExpBase(abc.ABC):
    """Base experiment class.

    Subclasses must implement:
      - ``_build_model()``
      - ``_get_data(split)``
      - ``train()``
      - ``test()``
    """

    def __init__(self, args) -> None:
        self.args = args
        self.device: torch.device = args.device
        self.model: nn.Module = self._build_model().to(self.device)
        # Populated by train() — one dict per epoch.  Read by run.py after training.
        self.epoch_logs: list[dict] = []

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _build_model(self) -> nn.Module:
        """Instantiate and return the model."""

    @abc.abstractmethod
    def _get_data(self, split: str):
        """Return (dataset, dataloader) for the given split."""

    @abc.abstractmethod
    def train(self):
        """Run the full training loop."""

    @abc.abstractmethod
    def test(self) -> dict:
        """Evaluate on the test set and return a metrics dict."""

    # ------------------------------------------------------------------
    def _get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

    def _get_criterion(self) -> nn.Module:
        return nn.MSELoss()
