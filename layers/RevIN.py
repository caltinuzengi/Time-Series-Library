"""Reversible Instance Normalization (RevIN).

Reference: Kim et al., ICLR 2022 — "Reversible Instance Normalization for
Accurate Time-Series Forecasting against Distribution Shift".
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization.

    Normalizes over the **time** dimension (dim=1) so each instance has
    zero mean and unit variance. Optionally applies learnable affine
    parameters after normalization (and inverts them before denormalization).

    Args:
        num_features: Number of variates (channels) C.
        eps: Small constant added to std for numerical stability.
        affine: If True, add learnable per-channel scale (γ) and shift (β).
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            # Shape (1, 1, C) for broadcasting over (B, T, C)
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

        # Cached statistics filled during 'norm', used during 'denorm'
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply normalization or denormalization.

        Args:
            x:    Input tensor of shape ``(B, T, C)``.
            mode: ``'norm'`` or ``'denorm'``.

        Returns:
            Transformed tensor of the same shape.
        """
        if mode == "norm":
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")

    # ------------------------------------------------------------------
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Compute and cache instance statistics over T (dim=1)
        self._mean = x.mean(dim=1, keepdim=True)          # (B, 1, C)
        self._std = x.std(dim=1, keepdim=True, unbiased=False) + self.eps  # (B, 1, C)

        x_hat = (x - self._mean) / self._std

        if self.affine:
            x_hat = x_hat * self.affine_weight + self.affine_bias

        return x_hat

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            raise RuntimeError("RevIN.forward('norm') must be called before 'denorm'.")

        if self.affine:
            # Invert affine first, then invert the standardization
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)

        return x * self._std + self._mean
