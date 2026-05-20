"""ModernTCN — Modern Pure Convolution Structure for Time Series Forecasting.

Reference:
    Luo & Wang, "ModernTCN: A Modern Pure Convolution Structure for General
    Time Series Analysis", ICLR 2024. https://openreview.net/forum?id=vpJMJerXHU
    Official code: https://github.com/luodhhh/ModernTCN
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from layers.ModernTCNBlock import ModernTCNBlock
from layers.RevIN import RevIN


class ModernTCN(nn.Module):
    """ModernTCN forecasting model.

    Architecture overview:

    1. RevIN normalisation: ``(B, T, C)``
    2. Stem (Conv1d, channel-independent):
       ``(B, C, T)`` → reshape ``(B*C, 1, T)``
       → Conv1d(1, d_model, patch_size, stride=patch_stride) + BatchNorm1d
       → reshape ``(B, C, d_model, N)``
       where ``N = seq_len // patch_stride``  (non-overlapping default)
    3. ``e_layers`` × :class:`ModernTCNBlock`:
       Each block operates on ``(B, M, D, N)`` and contains:

       * Structural-reparam DWConv (large + small parallel kernels) + BatchNorm1d
       * ConvFFN-1: per-variate feature mixing (groups = M)
       * ConvFFN-2: cross-variate mixing per feature (groups = D)
       * Single residual over the entire block

    4. FlattenHead:
       ``(B, C, D, N)`` → flatten last two dims → ``(B, C, D*N)``
       → Linear(D*N, pred_len) → ``(B, C, pred_len)``
       → permute → ``(B, pred_len, C)``
    5. RevIN denormalisation.

    Args:
        configs: Namespace with the following attributes::

            seq_len, pred_len       — sequence / forecast lengths
            enc_in, c_out           — input / output channels (= M)
            d_model, e_layers       — embedding dim / number of blocks
            patch_size              — Conv1d stem kernel size  (default 8)
            patch_stride            — Conv1d stem stride       (default 8)
            large_kernel            — DWConv large kernel size (default 51)
            small_kernel            — DWConv small kernel size (default 5)
            ffn_ratio               — ConvFFN expansion ratio  (default 4)
            dropout                 — dropout probability
    """

    def __init__(self, configs: SimpleNamespace) -> None:
        super().__init__()

        self.pred_len    = configs.pred_len
        self.enc_in      = configs.enc_in      # M
        self.patch_size  = configs.patch_size
        self.patch_stride = configs.patch_stride

        # Stem: each variate independently patched
        # Conv1d(1, d_model, patch_size, stride=patch_stride)
        self.stem = nn.Sequential(
            nn.Conv1d(
                1, configs.d_model,
                kernel_size=configs.patch_size,
                stride=configs.patch_stride,
                bias=False,
            ),
            nn.BatchNorm1d(configs.d_model),
        )

        # Number of patches produced by the stem
        if configs.patch_size != configs.patch_stride:
            # End-padding will add (patch_size - patch_stride) timesteps
            pad_len = configs.patch_size - configs.patch_stride
            num_patches = (configs.seq_len + pad_len - configs.patch_size) // configs.patch_stride + 1
        else:
            num_patches = configs.seq_len // configs.patch_stride
        self.num_patches  = num_patches
        self._need_pad    = configs.patch_size != configs.patch_stride
        self._pad_len     = configs.patch_size - configs.patch_stride if self._need_pad else 0

        # Encoder blocks
        self.blocks = nn.ModuleList(
            [
                ModernTCNBlock(
                    d_model=configs.d_model,
                    nvars=configs.enc_in,
                    large_kernel=configs.large_kernel,
                    small_kernel=configs.small_kernel,
                    ffn_ratio=configs.ffn_ratio,
                    dropout=configs.dropout,
                )
                for _ in range(configs.e_layers)
            ]
        )

        # Prediction head — shared linear across all variates
        head_nf = configs.d_model * num_patches
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),           # (B, M, D*N)
            nn.Dropout(configs.dropout),
            nn.Linear(head_nf, configs.pred_len),
        )

        self.seq_len   = configs.seq_len
        self._head_nf  = head_nf

        # Reconstruction head for anomaly detection (same backbone, seq_len output)
        self.recon_head = nn.Linear(head_nf, configs.seq_len)

        # Instance normalisation
        self.revin = RevIN(configs.enc_in, affine=True)

    # ------------------------------------------------------------------

    def _apply_stem(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stem conv to each variate independently.

        Args:
            x: ``(B, C, T)``

        Returns:
            ``(B, C, d_model, N)``
        """
        B, C, T = x.shape

        # Channel-independent reshape: (B, C, T) → (B*C, 1, T)
        x = x.reshape(B * C, 1, T)

        # Optional end-padding when patch_size > patch_stride
        if self._need_pad:
            pad = x[:, :, -1:].expand(-1, -1, self._pad_len)
            x = torch.cat([x, pad], dim=-1)     # (B*C, 1, T + pad_len)

        # Stem Conv1d: (B*C, 1, T) → (B*C, d_model, N)
        x = self.stem(x)

        _, D, N = x.shape
        return x.reshape(B, C, D, N)            # (B, C, d_model, N)

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
            x_mark_enc: ``(B, seq_len, 4)`` — time features (not used directly)
            x_dec:      Ignored (encoder-only model).
            x_mark_dec: Ignored.

        Returns:
            ``(B, pred_len, c_out)``
        """
        # 1. RevIN normalize: (B, T, C)
        x = self.revin(x_enc, "norm")

        # 2. Stem: (B, T, C) → (B, C, T) → (B, C, d_model, N)
        x = self._apply_stem(x.permute(0, 2, 1))   # permute (B,T,C)→(B,C,T)

        # 3. Encoder blocks: (B, C, d_model, N) → (B, C, d_model, N)
        for block in self.blocks:
            x = block(x)

        # 4. FlattenHead: (B, C, d_model, N) → (B, C, pred_len)
        x = self.head(x)                             # (B, C, pred_len)

        # 5. (B, C, pred_len) → (B, pred_len, C)
        x = x.permute(0, 2, 1).contiguous()

        # 6. RevIN denormalize
        x = self.revin(x, "denorm")

        return x

    # ------------------------------------------------------------------

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input for anomaly detection.

        Same stem + blocks as :meth:`forward`, but uses ``recon_head``
        to project to ``seq_len`` instead of ``pred_len``.

        Args:
            x_enc: ``(B, seq_len, enc_in)``

        Returns:
            reconstruction: ``(B, seq_len, enc_in)``
        """
        # 1. RevIN normalize
        x = self.revin(x_enc, "norm")                      # (B, T, C)

        # 2. Stem: (B, C, T) → (B, C, d_model, N)
        x = self._apply_stem(x.permute(0, 2, 1))

        # 3. Encoder blocks
        for block in self.blocks:
            x = block(x)                                   # (B, C, d_model, N)

        # 4. Flatten + reconstruct
        B, C, D, N = x.shape
        x = x.flatten(start_dim=-2)                        # (B, C, D*N)
        x = self.recon_head(x)                             # (B, C, seq_len)
        x = x.permute(0, 2, 1).contiguous()                # (B, seq_len, C)

        # 5. RevIN denormalize
        x = self.revin(x, "denorm")
        return x

    # ------------------------------------------------------------------

    def merge_reparams(self) -> None:
        """Merge structural-reparam DWConv kernels for efficient inference.

        Call this once after training.  Subsequent forward passes use a single
        fused Conv1d instead of two parallel paths.
        """
        for m in self.modules():
            if hasattr(m, "merge_kernel"):
                m.merge_kernel()
