"""PatchTST — Patch-based Transformer for Long-term Time Series Forecasting.

Reference:
    Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with
    Transformers", ICLR 2023. https://arxiv.org/abs/2211.14730
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from layers.PatchEmbedding import PatchEmbedding
from layers.RevIN import RevIN


# ---------------------------------------------------------------------------
# Transformer encoder block
# ---------------------------------------------------------------------------

class _TSTEncoderLayer(nn.Module):
    """Single pre-norm Transformer encoder layer.

    Pre-norm (norm_first=True) applies LayerNorm *before* each sub-layer,
    which is more stable than the original post-norm variant for deep models.

    Sub-layers:
        1. LayerNorm → MultiheadAttention → residual → Dropout
        2. LayerNorm → FFN (d_model→d_ff→d_model) → residual → Dropout

    Uses ``nn.MultiheadAttention`` with ``batch_first=True`` and
    ``dropout`` applied inside attention weights.  When PyTorch 2.0+'s
    ``scaled_dot_product_attention`` is available, Flash Attention is used
    automatically.

    Args:
        d_model:  Embedding dimension.
        n_heads:  Number of attention heads.  Must divide ``d_model`` evenly.
        d_ff:     Hidden dimension of the feed-forward sub-layer.
        dropout:  Dropout probability applied after each sub-layer.

    Shape:
        input / output: ``(B, N, d_model)``
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, N, d_model)``

        Returns:
            ``(B, N, d_model)``
        """
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.drop1(attn_out)

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))

        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PatchTST(nn.Module):
    """PatchTST forecasting model (channel-independent mode).

    Architecture overview:

    1. RevIN normalisation: ``(B, T, C)``
    2. PatchEmbedding (channel-independent):
       ``(B, T, C)`` → ``(B*C, N, d_model)``
       where ``N = (T - patch_len) // stride + 1 [+1 with end-padding]``
    3. ``e_layers`` × :class:`_TSTEncoderLayer` (pre-norm Transformer)
    4. FlattenHead:
       ``(B*C, N, d_model)`` → flatten → ``(B*C, N*d_model)``
       → ``Linear(N*d_model, pred_len)`` → ``(B*C, pred_len)``
       → reshape → ``(B, pred_len, C)``
    5. RevIN denormalisation.

    Args:
        configs: Namespace with the following attributes::

            seq_len, pred_len       — sequence / forecast lengths
            enc_in, c_out           — input / output channels
            d_model, d_ff, e_layers — model width / depth
            n_heads                 — attention heads
            patch_len               — patch length         (default 16)
            stride                  — patch stride         (default 8)
            dropout                 — dropout probability
    """

    def __init__(self, configs: SimpleNamespace) -> None:
        super().__init__()

        self.pred_len = configs.pred_len
        self.enc_in   = configs.enc_in

        # --- Instance normalisation ---
        self.revin = RevIN(configs.enc_in, affine=True)

        # --- Patch embedding ---
        self.patch_embed = PatchEmbedding(
            seq_len=configs.seq_len,
            patch_len=configs.patch_len,
            stride=configs.stride,
            d_model=configs.d_model,
            dropout=configs.dropout,
            padding_patch="end",
        )
        num_patches = self.patch_embed.num_patches

        # --- Transformer encoder ---
        self.encoder = nn.ModuleList(
            [
                _TSTEncoderLayer(
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(configs.d_model)

        self.seq_len     = configs.seq_len
        self.num_patches = num_patches
        self._d_model    = configs.d_model

        # --- Prediction head ---
        # Flatten all patch tokens, then project to pred_len
        self.head = nn.Linear(num_patches * configs.d_model, configs.pred_len)

        # Reconstruction head for anomaly detection (same backbone, seq_len output)
        self.recon_head = nn.Linear(num_patches * configs.d_model, configs.seq_len)

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
            x_mark_enc: ``(B, seq_len, 4)`` — time features (not used by PatchTST)
            x_dec:      Ignored (encoder-only model).
            x_mark_dec: Ignored.

        Returns:
            ``(B, pred_len, c_out)``
        """
        B, _T, C = x_enc.shape

        # 1. RevIN normalize
        x = self.revin(x_enc, "norm")            # (B, T, C)

        # 2. Patch embedding — channel-independent
        x = self.patch_embed(x)                  # (B*C, N, d_model)

        # 3. Transformer encoder
        for layer in self.encoder:
            x = layer(x)                         # (B*C, N, d_model)
        x = self.encoder_norm(x)

        # 4. Flatten + predict
        x = x.flatten(start_dim=1)               # (B*C, N*d_model)
        x = self.head(x)                         # (B*C, pred_len)

        # Reshape to (B, pred_len, C)
        x = (
            x.reshape(B, C, self.pred_len)       # (B, C, pred_len)
            .permute(0, 2, 1)                    # (B, pred_len, C)
            .contiguous()
        )

        # 5. RevIN denormalize
        x = self.revin(x, "denorm")

        return x

    # ------------------------------------------------------------------

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input sequence for anomaly detection.

        Same encoder backbone as :meth:`forward`, but uses ``recon_head``
        to project to ``seq_len`` instead of ``pred_len``.

        Args:
            x_enc: ``(B, seq_len, enc_in)``

        Returns:
            reconstruction: ``(B, seq_len, enc_in)``
        """
        B, T, C = x_enc.shape

        # 1. RevIN normalize
        x = self.revin(x_enc, "norm")            # (B, T, C)

        # 2. Patch embedding — channel-independent
        x = self.patch_embed(x)                  # (B*C, N, d_model)

        # 3. Transformer encoder
        for layer in self.encoder:
            x = layer(x)                         # (B*C, N, d_model)
        x = self.encoder_norm(x)

        # 4. Flatten + reconstruct
        x = x.flatten(start_dim=1)               # (B*C, N*d_model)
        x = self.recon_head(x)                   # (B*C, seq_len)

        x = (
            x.reshape(B, C, T)                   # (B, C, T)
            .permute(0, 2, 1)                    # (B, T, C)
            .contiguous()
        )

        # 5. RevIN denormalize
        x = self.revin(x, "denorm")
        return x
