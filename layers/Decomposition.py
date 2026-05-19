"""Decomposition layers for Time Series Library.

MovingAverage and SeriesDecomposition used by TimeMixer.

Reference:
    Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series
    Forecasting", ICLR 2024. https://openreview.net/forum?id=7oLshfEIC2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    """1D moving average filter for trend extraction.

    Applies ``avg_pool1d`` with replicate edge padding so that the output
    sequence length equals the input sequence length.

    Args:
        kernel_size: Window size.  Odd values give symmetric padding.

    Shape:
        input / output: ``(B, T, C)``
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, C)``

        Returns:
            ``(B, T, C)`` — smoothed signal.
        """
        # Replicate-pad both ends to keep output length == T after stride-1 pool
        front = x[:, :1, :].expand(-1, (self.kernel_size - 1) // 2, -1)
        back  = x[:, -1:, :].expand(-1, self.kernel_size // 2, -1)
        x_pad = torch.cat([front, x, back], dim=1)            # (B, T+k-1, C)

        # avg_pool1d expects (B, C, T)
        out = F.avg_pool1d(
            x_pad.permute(0, 2, 1),                           # (B, C, T+k-1)
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )                                                      # (B, C, T)
        return out.permute(0, 2, 1)                           # (B, T, C)


class SeriesDecomposition(nn.Module):
    """Separates a time series into trend and seasonal components.

    ``trend    = MovingAverage(x)``
    ``seasonal = x - trend``

    Round-trip identity: ``seasonal + trend == x`` (exact, no numerical drift).

    Args:
        kernel_size: Moving-average window size.

    Shape:
        input:  ``(B, T, C)``
        output: ``(seasonal, trend)`` — each ``(B, T, C)``
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(seasonal, trend)`` — both ``(B, T, C)``."""
        trend    = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend
