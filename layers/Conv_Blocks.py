"""Convolutional building blocks for Time Series Library.

Blocks implemented here:
  - InceptionBlock  — multi-scale 2-D inception module (TimesNet)
  - TimesBlock      — FFT-based temporal 2-D representation block (TimesNet)

Reference:
  Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series
  Analysis", ICLR 2023.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# InceptionBlock
# ---------------------------------------------------------------------------

class InceptionBlock(nn.Module):
    """Multi-scale 2-D inception block.

    Applies ``num_kernels`` parallel Conv2d layers with kernel widths
    ``[1, 3, 5, 7, 9, 11]`` (same-padding) then averages their outputs.

    All convolution weights are Kaiming-normal initialised.

    Args:
        in_channels:  Number of input channels (== d_model after reshape).
        out_channels: Number of output channels.
        num_kernels:  How many kernels to use (max 6, uses first N from the list).
        init_weight:  If True, apply Kaiming-normal init.

    Shape:
        input:  ``(B, C, H, W)``
        output: ``(B, C, H, W)``  (same spatial size via padding)
    """

    # Kernel sizes as defined in the paper (Eq. 5)
    _KERNEL_SIZES = [1, 3, 5, 7, 9, 11]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        init_weight: bool = True,
    ) -> None:
        super().__init__()
        kernels = self._KERNEL_SIZES[:num_kernels]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, k),
                    padding=(0, k // 2),  # same padding along W
                )
                for k in kernels
            ]
        )
        if init_weight:
            self._init_weights()

    def _init_weights(self) -> None:
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, C_in, H, W)``

        Returns:
            ``(B, C_out, H, W)``
        """
        outs = torch.stack([conv(x) for conv in self.convs], dim=-1)  # (B,C,H,W,K)
        return outs.mean(dim=-1)                                        # (B,C,H,W)


# ---------------------------------------------------------------------------
# TimesBlock
# ---------------------------------------------------------------------------

class TimesBlock(nn.Module):
    """Temporal 2-D Variation block (TimesNet backbone unit).

    Steps (paper Algorithm 1):
      1. rfft over the time axis → amplitudes → top-k dominant frequencies
         (DC component zeroed out).
      2. For each period p derived from the top-k frequencies:
           a. Pad the sequence to the nearest multiple of p.
           b. Reshape 1-D sequence → 2-D grid (B·N, 1, T//p, p).
           c. Apply the **shared** InceptionBlock.
           d. Reshape back and truncate to original length T.
      3. Softmax-weighted sum of all k residuals.
      4. Add residual connection.

    Args:
        seq_len:     Encoder input length L.
        pred_len:    Forecast horizon H.
        d_model:     Model (embedding) dimension.
        d_ff:        InceptionBlock output channels (== d_model here; kept
                     as a distinct hyper-parameter for future flexibility).
        top_k:       Number of dominant frequencies / periods to consider.
        num_kernels: Passed to InceptionBlock.

    Shape:
        input/output: ``(B, T, d_model)``  where T = seq_len + pred_len.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        d_ff: int,
        top_k: int = 5,
        num_kernels: int = 6,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.top_k = top_k

        # Two-stage InceptionBlock: d_model → d_ff (GELU) → d_model
        # Matches TSLib reference implementation.
        # Reshape keeps batch=B, not B*d_model, for GPU efficiency.
        self.conv = nn.Sequential(
            InceptionBlock(in_channels=d_model, out_channels=d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlock(in_channels=d_ff, out_channels=d_model, num_kernels=num_kernels),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _top_k_periods(x: torch.Tensor, top_k: int) -> tuple[torch.Tensor, list[int]]:
        """Compute top-k dominant periods via real FFT.

        Returns:
            weights: Softmax weights for each period, shape ``(top_k,)``.
            periods: List of integer period values.
        """
        T = x.shape[1]
        # rfft over time → (B, T//2+1, N) complex
        xf = torch.fft.rfft(x, dim=1)
        # Mean amplitude over batch and variates → (T//2+1,)
        amp = xf.abs().mean(dim=(0, 2))
        amp[0] = 0.0  # zero DC component

        # Select top-k frequencies (indices ≥ 1)
        k = min(top_k, amp.shape[0] - 1)
        _, top_idx = torch.topk(amp[1:], k)
        top_idx = top_idx + 1  # shift back (index 0 == DC)

        # Convert frequency index → period length; clamp to ≥ 1
        periods = [max(1, T // idx.item()) for idx in top_idx]

        weights = F.softmax(amp[top_idx], dim=0)  # (k,)
        return weights, periods

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, d_model)``  T = seq_len + pred_len

        Returns:
            ``(B, T, d_model)``
        """
        B, T, N = x.shape

        weights, periods = self._top_k_periods(x, self.top_k)

        residuals: list[torch.Tensor] = []

        for period in periods:
            # Pad time dim to nearest multiple of period
            pad_len = math.ceil(T / period) * period - T
            x_pad = F.pad(x, (0, 0, 0, pad_len)) if pad_len > 0 else x
            rows = x_pad.shape[1] // period

            # 1-D → 2-D: (B, rows*period, N) → (B, N, rows, period)
            # Keeps batch=B (not B*N) — correct cross-channel mixing, 64x less batch overhead
            x_2d = rearrange(x_pad, "b (r p) n -> b n r p", p=period)

            # Two-stage conv: (B, N, rows, period) → (B, N, rows, period)
            out_2d = self.conv(x_2d)

            # 2-D → 1-D: (B, N, rows, period) → (B, rows*period, N) → truncate
            out_1d = rearrange(out_2d, "b n r p -> b (r p) n")
            residuals.append(out_1d[:, :T, :])

        # Softmax-weighted aggregation + residual
        out = sum(w * r for w, r in zip(weights, residuals))   # (B, T, N)
        return out + x
