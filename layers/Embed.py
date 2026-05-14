"""Embedding layers for Time Series Library.

DataEmbedding: sum of a linear value projection and a linear time-feature
projection, followed by dropout.  Used by TimesNet (and later TimeMixer).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DataEmbedding(nn.Module):
    """Value + temporal embedding with dropout.

    Args:
        c_in:    Number of input variates (channels).
        d_model: Embedding / hidden dimension.
        dropout: Dropout probability.

    Shape:
        x:      ``(B, T, c_in)``
        x_mark: ``(B, T, 4)``  — time features from ``time_features()``
        output: ``(B, T, d_model)``
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        # bias=False follows the original TimesNet implementation
        self.value_embedding = nn.Linear(c_in, d_model, bias=False)
        self.temporal_embedding = nn.Linear(4, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      ``(B, T, c_in)``
            x_mark: ``(B, T, 4)``

        Returns:
            ``(B, T, d_model)``
        """
        out = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(out)
