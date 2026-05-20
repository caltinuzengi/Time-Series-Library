"""TimeMixer — Decomposable Multiscale Mixing for Time Series Forecasting.

Reference:
    Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series
    Forecasting", ICLR 2024. https://openreview.net/forum?id=7oLshfEIC2
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding
from layers.Mixing import PastDecomposableMixing
from layers.RevIN import RevIN


class TimeMixer(nn.Module):
    """TimeMixer forecasting model (channel-independent mode).

    Architecture overview:

    1. Multiscale downsampling:
       Create ``down_sampling_layers + 1`` scales via avg_pool1d, so
       ``T_i = seq_len // (down_sampling_window ** i)``.

    2. Per-scale RevIN normalisation + channel-independent reshape:
       ``(B, T_i, C) → normalize → (B*C, T_i, 1)``

    3. DataEmbedding per scale (c_in=1):
       ``(B*C, T_i, 1) → (B*C, T_i, d_model)``

    4. ``e_layers`` × PastDecomposableMixing blocks (PDM).

    5. Future Multipredictor Mixing (FMM):
       For each scale ``i``: ``Linear(T_i, pred_len)`` along the time axis,
       then ``Linear(d_model, 1)`` → reshape to ``(B, pred_len, C)``.
       All scales are **summed** (no learned weighting).

    6. RevIN denormalisation using scale-0 statistics.

    Args:
        configs: Namespace with the following attributes::

            seq_len, pred_len          — sequence / forecast lengths
            enc_in, c_out              — input / output channels
            d_model, d_ff, e_layers    — transformer widths / depth
            dropout                    — dropout probability
            down_sampling_layers       — number of downsampling steps  (default 3)
            down_sampling_window       — downsampling factor            (default 2)
            moving_avg                 — MA kernel size for decomp      (default 25)
    """

    def __init__(self, configs: SimpleNamespace) -> None:
        super().__init__()

        self.seq_len              = configs.seq_len
        self.pred_len             = configs.pred_len
        self.enc_in               = configs.enc_in
        self.down_sampling_layers = configs.down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window

        n_scales = configs.down_sampling_layers + 1

        # Per-scale instance normalisation (one RevIN per scale)
        self.normalize_layers = nn.ModuleList(
            [RevIN(configs.enc_in, affine=True) for _ in range(n_scales)]
        )

        # Input embedding — channel-independent: always c_in=1
        self.enc_embedding = DataEmbedding(
            c_in=1,
            d_model=configs.d_model,
            dropout=configs.dropout,
        )

        # PDM encoder blocks
        self.pdm_blocks = nn.ModuleList(
            [
                PastDecomposableMixing(
                    seq_len=configs.seq_len,
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    down_sampling_layers=configs.down_sampling_layers,
                    down_sampling_window=configs.down_sampling_window,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                )
                for _ in range(configs.e_layers)
            ]
        )

        # Per-scale temporal predictor: Linear(T_i, pred_len)
        self.predict_layers = nn.ModuleList(
            [
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(n_scales)
            ]
        )

        # Final channel projection: d_model → 1 (channel-independent)
        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)

    # ------------------------------------------------------------------

    def _multiscale_inputs(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Build multi-scale versions of ``x_enc`` and ``x_mark_enc``.

        Returns:
            x_list:    list of ``(B, T_i, C)`` tensors, finest scale first.
            mark_list: list of ``(B, T_i, 4)`` tensors, finest scale first.
        """
        x_list    = [x_enc]
        mark_list = [x_mark_enc]

        # avg_pool1d expects (B, C, T)
        x_cur    = x_enc.permute(0, 2, 1)
        mark_cur = x_mark_enc

        for _ in range(self.down_sampling_layers):
            x_cur = F.avg_pool1d(
                x_cur,
                kernel_size=self.down_sampling_window,
                stride=self.down_sampling_window,
                padding=0,
            )                                           # (B, C, T_i)
            x_list.append(x_cur.permute(0, 2, 1))      # (B, T_i, C)

            # Subsample marks along time (no averaging for feature vectors)
            mark_cur = mark_cur[:, :: self.down_sampling_window, :]
            mark_list.append(mark_cur)

        return x_list, mark_list

    # ------------------------------------------------------------------

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_enc:      ``(B, seq_len, enc_in)``
            x_mark_enc: ``(B, seq_len, 4)``
            x_dec:      Ignored (encoder-only model).
            x_mark_dec: Ignored.

        Returns:
            ``(B, pred_len, c_out)``
        """
        B, _T, C = x_enc.size()

        # 1. Multi-scale downsampling
        x_list, mark_list = self._multiscale_inputs(x_enc, x_mark_enc)

        # 2. Per-scale: RevIN norm → channel-independent reshape → embed
        enc_out_list: list[torch.Tensor] = []
        for i, (x_scale, mark_scale) in enumerate(zip(x_list, mark_list)):
            B_i, T_i, _ = x_scale.size()

            # RevIN normalise: (B, T_i, C) → (B, T_i, C)
            x_scale = self.normalize_layers[i](x_scale, "norm")

            # Channel-independent reshape: (B, T_i, C) → (B*C, T_i, 1)
            x_flat = (
                x_scale.permute(0, 2, 1)          # (B, C, T_i)
                .contiguous()
                .reshape(B_i * C, T_i, 1)         # (B*C, T_i, 1)
            )

            # Repeat marks for all channels: (B, T_i, 4) → (B*C, T_i, 4)
            # repeat_interleave preserves the batch alignment with x_flat
            mark_flat = mark_scale.repeat_interleave(C, dim=0)  # (B*C, T_i, 4)

            # DataEmbedding: (B*C, T_i, 1) → (B*C, T_i, d_model)
            enc_out = self.enc_embedding(x_flat, mark_flat)
            enc_out_list.append(enc_out)

        # 3. PDM encoder blocks
        for pdm_block in self.pdm_blocks:
            enc_out_list = pdm_block(enc_out_list)

        # 4. Future Multipredictor Mixing (FMM) + sum
        dec_out_list: list[torch.Tensor] = []
        for i, enc_out in enumerate(enc_out_list):
            # enc_out: (B*C, T_i, d_model)

            # Predict along time axis: permute → Linear(T_i, pred_len) → permute
            dec = self.predict_layers[i](
                enc_out.permute(0, 2, 1)           # (B*C, d_model, T_i)
            ).permute(0, 2, 1)                     # (B*C, pred_len, d_model)

            # Project to channel: (B*C, pred_len, 1)
            dec = self.projection_layer(dec)

            # Reshape to (B, pred_len, C)
            dec = (
                dec.reshape(B, C, self.pred_len)   # (B, C, pred_len)
                .permute(0, 2, 1)                  # (B, pred_len, C)
                .contiguous()
            )
            dec_out_list.append(dec)

        # Sum across scales — plain sum, no softmax weighting
        dec_out = torch.stack(dec_out_list, dim=-1).sum(dim=-1)  # (B, pred_len, C)

        # 5. Denormalise with scale-0 statistics
        dec_out = self.normalize_layers[0](dec_out, "denorm")

        return dec_out

    # ------------------------------------------------------------------

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input for anomaly detection.

        Uses the finest-scale PDM embedding directly, bypassing
        ``predict_layers`` (which are future-oriented).
        Reuses ``projection_layer`` for the channel projection.

        Args:
            x_enc: ``(B, seq_len, enc_in)``

        Returns:
            reconstruction: ``(B, seq_len, enc_in)``
        """
        B, T, C = x_enc.shape

        # 1. Multi-scale downsampling (marks ignored — see note below)
        # We pass a dummy zero mark; _multiscale_inputs uses ::stride indexing for
        # marks but avg_pool1d (floor division) for x, so their T_i can differ for
        # seq_len values that are not divisible by down_sampling_window^n.
        # To avoid the shape mismatch we create per-scale zero marks directly.
        dummy_mark = torch.zeros(B, T, 4, device=x_enc.device, dtype=x_enc.dtype)
        x_list, _ = self._multiscale_inputs(x_enc, dummy_mark)

        # 2. Per-scale: RevIN norm → channel-independent reshape → embed
        enc_out_list: list[torch.Tensor] = []
        for i, x_scale in enumerate(x_list):
            _, T_i, _ = x_scale.size()
            x_scale = self.normalize_layers[i](x_scale, "norm")
            x_flat = (
                x_scale.permute(0, 2, 1)             # (B, C, T_i)
                .contiguous()
                .reshape(B * C, T_i, 1)              # (B*C, T_i, 1)
            )
            # Zero marks with correct T_i length (no temporal metadata in SMD)
            mark_flat = torch.zeros(
                B * C, T_i, 4, device=x_enc.device, dtype=x_enc.dtype
            )
            enc_out_list.append(self.enc_embedding(x_flat, mark_flat))

        # 3. PDM encoder blocks
        for pdm_block in self.pdm_blocks:
            enc_out_list = pdm_block(enc_out_list)

        # 4. Reconstruct from finest-scale embedding (bypasses predict_layers)
        enc_0 = enc_out_list[0]                          # (B*C, T, d_model)
        dec   = self.projection_layer(enc_0)             # (B*C, T, 1)
        dec   = (
            dec.reshape(B, C, T)                         # (B, C, T)
            .permute(0, 2, 1)                            # (B, T, C)
            .contiguous()
        )

        # 5. RevIN denorm using scale-0 statistics
        dec = self.normalize_layers[0](dec, "denorm")
        return dec
