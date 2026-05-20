"""Mixing layers for TimeMixer.

Components:
    MultiScaleSeasonMixing  — bottom-up seasonal mixing (fine → coarse)
    MultiScaleTrendMixing   — top-down trend mixing    (coarse → fine)
    PastDecomposableMixing  — full PDM block (decompose + mix + residual)

Reference:
    Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series
    Forecasting", ICLR 2024. https://openreview.net/forum?id=7oLshfEIC2
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.Decomposition import SeriesDecomposition


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up seasonal mixing across scales (fine → coarse).

    For consecutive scales ``i`` and ``i+1``, a two-layer MLP projects the
    seasonal component of the finer scale to the coarser scale and adds it
    as a residual.

    Each layer maps along the time axis (last dim after permute):
        ``(B, d_model, T_i) → (B, d_model, T_{i+1})``
    where ``T_{i+1} = seq_len // (down_sampling_window ** (i+1))``.

    Args:
        seq_len:              Original (finest) sequence length.
        d_model:              Embedding dimension.
        down_sampling_layers: Number of downsampling steps (= n_scales - 1).
        down_sampling_window: Downsampling factor between consecutive scales.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        down_sampling_layers: int,
        down_sampling_window: int,
    ) -> None:
        super().__init__()
        self.down_sampling_layers = down_sampling_layers
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            season_list: List of ``(B, d_model, T_i)`` tensors, finest first.

        Returns:
            List of ``(B, T_i, d_model)`` tensors, finest first.
        """
        out_high = season_list[0]
        out_list = [out_high.permute(0, 2, 1)]   # (B, T_0, d_model)

        if len(season_list) == 1:
            return out_list

        out_low = season_list[1]
        for i in range(len(season_list) - 1):
            out_low_res = self.layers[i](out_high)   # (B, d_model, T_{i+1})
            out_low     = out_low + out_low_res
            out_high    = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_list.append(out_high.permute(0, 2, 1))  # (B, T_{i+1}, d_model)

        return out_list


class MultiScaleTrendMixing(nn.Module):
    """Top-down trend mixing across scales (coarse → fine).

    For consecutive scales, a two-layer MLP upsamples the trend component
    from the coarser scale to the finer scale and adds it as a residual.

    Each layer maps along the time axis:
        ``(B, d_model, T_{i+1}) → (B, d_model, T_i)``
    Layers are stored coarsest-first to match the forward iteration order.

    Args:
        seq_len:              Original (finest) sequence length.
        d_model:              Embedding dimension.
        down_sampling_layers: Number of downsampling steps (= n_scales - 1).
        down_sampling_window: Downsampling factor between consecutive scales.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        down_sampling_layers: int,
        down_sampling_window: int,
    ) -> None:
        super().__init__()
        self.down_sampling_layers = down_sampling_layers
        # reversed(range(n)) gives n-1, n-2, ..., 0 so layer[0] maps
        # T_n → T_{n-1}, layer[1] maps T_{n-1} → T_{n-2}, etc.
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ]
        )

    def forward(self, trend_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            trend_list: List of ``(B, d_model, T_i)`` tensors, finest first.

        Returns:
            List of ``(B, T_i, d_model)`` tensors, finest first.
        """
        trend_rev = list(reversed(trend_list))   # coarsest first
        out_low   = trend_rev[0]
        out_list  = [out_low.permute(0, 2, 1)]   # (B, T_n, d_model)

        if len(trend_rev) == 1:
            out_list.reverse()
            return out_list

        out_high = trend_rev[1]
        for i in range(len(trend_rev) - 1):
            out_high_res = self.layers[i](out_low)   # (B, d_model, T_{finer})
            out_high     = out_high + out_high_res
            out_low      = out_high
            if i + 2 <= len(trend_rev) - 1:
                out_high = trend_rev[i + 2]
            out_list.append(out_low.permute(0, 2, 1))  # (B, T_{finer}, d_model)

        out_list.reverse()   # restore finest-first order
        return out_list


class PastDecomposableMixing(nn.Module):
    """Past Decomposable Mixing (PDM) block — one encoder layer of TimeMixer.

    For a list of multi-scale encoded representations:

    1. Decompose each scale:  ``seasonal_i, trend_i = SeriesDecomposition(x_i)``
    2. Bottom-up seasonal mixing  (``MultiScaleSeasonMixing``).
    3. Top-down trend mixing      (``MultiScaleTrendMixing``).
    4. Residual:  ``out_i = x_i + cross_layer(mixed_season_i + mixed_trend_i)``

    Args:
        seq_len:              Original (finest) sequence length.
        d_model:              Embedding dimension.
        d_ff:                 Hidden dim in the cross-scale residual MLP.
        down_sampling_layers: Number of downsampling steps (= n_scales - 1).
        down_sampling_window: Downsampling factor between consecutive scales.
        moving_avg:           Moving-average kernel size for decomposition.
        dropout:              Dropout probability.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff: int,
        down_sampling_layers: int,
        down_sampling_window: int,
        moving_avg: int = 25,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.decomposition = SeriesDecomposition(moving_avg)
        self.layer_norm    = nn.LayerNorm(d_model)
        self.dropout       = nn.Dropout(dropout)

        self.season_mixing = MultiScaleSeasonMixing(
            seq_len=seq_len,
            d_model=d_model,
            down_sampling_layers=down_sampling_layers,
            down_sampling_window=down_sampling_window,
        )
        self.trend_mixing = MultiScaleTrendMixing(
            seq_len=seq_len,
            d_model=d_model,
            down_sampling_layers=down_sampling_layers,
            down_sampling_window=down_sampling_window,
        )

        # Cross-scale residual MLP
        self.out_cross_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            x_list: List of ``(B, T_i, d_model)`` tensors, finest first.

        Returns:
            List of ``(B, T_i, d_model)`` tensors, finest first.
        """
        season_list: list[torch.Tensor] = []
        trend_list:  list[torch.Tensor] = []
        length_list: list[int]          = []

        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)
            seasonal, trend = self.decomposition(x)
            # Permute to (B, d_model, T) for Linear-along-time mixing
            season_list.append(seasonal.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # Bottom-up seasonal mixing → each (B, T_i, d_model)
        out_season_list = self.season_mixing(season_list)

        # Top-down trend mixing → each (B, T_i, d_model)
        out_trend_list  = self.trend_mixing(trend_list)

        # Residual + cross-layer projection + LayerNorm
        out_list: list[torch.Tensor] = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            mixed = out_season + out_trend                              # (B, T_i, d_model)
            out   = self.layer_norm(ori + self.out_cross_layer(mixed)) # (B, T_i, d_model)
            out_list.append(out[:, :length, :])             # safety truncation

        return out_list
