"""Patch embedding layer for PatchTST.

Converts a multivariate time series into patch tokens using channel-independent
processing: each variate is split into overlapping/non-overlapping patches,
projected to d_model, and augmented with learnable positional encodings.

Reference:
    Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with
    Transformers", ICLR 2023. https://arxiv.org/abs/2211.14730
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Channel-independent patch tokenizer.

    Processing pipeline:
        1. Reshape ``(B, T, C)`` → ``(B*C, T)`` (channel-independent)
        2. Optional end-padding: repeat the last timestep ``stride`` times so
           the sequence fits an extra patch without information leakage.
        3. Unfold into patches: ``(B*C, T_padded)`` → ``(B*C, N, patch_len)``
        4. Linear projection: ``(B*C, N, patch_len)`` → ``(B*C, N, d_model)``
        5. Add learnable positional encoding ``(1, N, d_model)`` + Dropout

    Args:
        seq_len:       Input sequence length ``T``.
        patch_len:     Length of each patch.
        stride:        Step size between consecutive patches.
        d_model:       Output embedding dimension.
        dropout:       Dropout probability applied after positional encoding.
        padding_patch: ``'end'`` (default) pads by repeating the last value
                       ``stride`` times, yielding one extra patch.  ``None``
                       skips padding.

    Attributes:
        num_patches (int): Total number of patches ``N`` produced per variate.

    Shape:
        input:  ``(B, T, C)``
        output: ``(B*C, N, d_model)``
    """

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        dropout: float,
        padding_patch: str | None = "end",
    ) -> None:
        super().__init__()
        self.patch_len     = patch_len
        self.stride        = stride
        self.padding_patch = padding_patch

        # Number of patches (before optional padding)
        self.num_patches = (seq_len - patch_len) // stride + 1
        if padding_patch == "end":
            self.num_patches += 1   # one extra patch from the padded tail

        # Patch → embedding projection (no bias, consistent with ViT convention)
        self.proj = nn.Linear(patch_len, d_model, bias=False)

        # Learnable positional encoding — one vector per patch position
        self.pos_enc = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        self.dropout = nn.Dropout(p=dropout)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, C)``

        Returns:
            ``(B*C, N, d_model)``
        """
        B, T, C = x.shape

        # 1. Channel-independent reshape: (B, T, C) → (B*C, T)
        x = x.permute(0, 2, 1).reshape(B * C, T)   # (B*C, T)

        # 2. Optional end-padding: repeat last value `stride` times
        if self.padding_patch == "end":
            x = torch.cat(
                [x, x[:, -1:].expand(-1, self.stride)], dim=-1
            )   # (B*C, T + stride)

        # 3. Unfold into patches: (B*C, N, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # 4. Project: (B*C, N, d_model)
        x = self.proj(x)

        # 5. Positional encoding + dropout
        x = x + self.pos_enc
        return self.dropout(x)
