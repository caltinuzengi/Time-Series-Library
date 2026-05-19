"""TimesNet — Temporal 2D-Variation Model for Time Series Forecasting.

Reference:
  Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series
  Analysis", ICLR 2023.  https://openreview.net/forum?id=ju_Uqw384Oq
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from layers.Conv_Blocks import TimesBlock
from layers.Embed import DataEmbedding
from layers.RevIN import RevIN


class TimesNet(nn.Module):
    """TimesNet forecasting model.

    Forward pass:
      1. RevIN normalisation
      2. DataEmbedding  (value + temporal)
      3. Zero-pad right by pred_len to form the full temporal context
      4. e_layers × TimesBlock (each followed by LayerNorm)
      5. Take the last pred_len steps
      6. Linear projection → c_out
      7. RevIN denormalisation

    Args:
        configs: A ``SimpleNamespace`` (or any object with attributes) containing:

          ==================  =====  ========================================
          seq_len             int    Encoder input length
          pred_len            int    Forecast horizon
          label_len           int    Decoder overlap (unused in forward)
          enc_in              int    Number of input variates
          c_out               int    Number of output variates
          d_model             int    Embedding / model dimension
          d_ff                int    InceptionBlock output channels
          e_layers            int    Number of TimesBlocks
          top_k               int    FFT top-k periods
          num_kernels         int    InceptionBlock kernels
          dropout             float  Dropout probability
          ==================  =====  ========================================
    """

    def __init__(self, configs: SimpleNamespace) -> None:
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.e_layers = configs.e_layers

        # --- Instance normalisation ---
        self.revin = RevIN(configs.enc_in, affine=True)

        # --- Input embedding ---
        self.embedding = DataEmbedding(
            c_in=configs.enc_in,
            d_model=configs.d_model,
            dropout=configs.dropout,
        )

        # --- Learned temporal expansion: seq_len → seq_len + pred_len ---
        # Applied per-channel along the time axis (matches TSLib reference).
        self.predict_linear = nn.Linear(
            configs.seq_len, configs.pred_len + configs.seq_len
        )

        # --- TimesBlocks + LayerNorms ---
        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    seq_len=configs.seq_len,
                    pred_len=configs.pred_len,
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    top_k=configs.top_k,
                    num_kernels=configs.num_kernels,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)]
        )

        # --- Output projection ---
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor | None = None,       # kept for API compatibility
        x_mark_dec: torch.Tensor | None = None,  # kept for API compatibility
    ) -> torch.Tensor:
        """
        Args:
            x_enc:      ``(B, seq_len, enc_in)``
            x_mark_enc: ``(B, seq_len, 4)``
            x_dec:      Ignored (no decoder in TimesNet).
            x_mark_dec: Ignored.

        Returns:
            ``(B, pred_len, c_out)``
        """
        # 1. RevIN normalise
        x = self.revin(x_enc, "norm")                         # (B, L, C)

        # 2. Embed
        enc = self.embedding(x, x_mark_enc)                   # (B, L, d_model)

        # 3. Learned temporal expansion: seq_len → seq_len + pred_len
        #    permute time↔channel so Linear acts on the time axis
        enc = self.predict_linear(enc.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L+H, d_model)

        # 4. TimesBlocks
        for block, norm in zip(self.blocks, self.norms):
            enc = norm(block(enc))                             # (B, L+H, d_model)

        # 5. Take last pred_len steps
        dec = enc[:, -self.pred_len:, :]                      # (B, H, d_model)

        # 6. Project
        out = self.projection(dec)                             # (B, H, c_out)

        # 7. RevIN denormalise
        out = self.revin(out, "denorm")                        # (B, H, c_out)
        return out
