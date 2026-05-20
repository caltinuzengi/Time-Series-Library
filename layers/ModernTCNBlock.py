"""ModernTCN block components.

Implements the core building blocks of ModernTCN exactly as described in the
reference code (luodhhh/ModernTCN):

    _fuse_bn              — fuse BatchNorm into Conv1d weights (reparam helper)
    _ReparamDWConv        — structural-reparam depthwise conv (large + small paths)
    ModernTCNBlock        — full block: DWConv + BN + ConvFFN1 + ConvFFN2 + residual

Tensor convention throughout: ``(B, M, D, N)``
    B — batch size
    M — number of variates  (= enc_in)
    D — embedding dimension (= d_model)
    N — number of patches   (= seq_len // patch_stride)

Reference:
    Luo & Wang, "ModernTCN: A Modern Pure Convolution Structure for General
    Time Series Analysis", ICLR 2024. https://openreview.net/forum?id=vpJMJerXHU
    Official code: https://github.com/luodhhh/ModernTCN
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# BN-fusion helper
# ---------------------------------------------------------------------------

def _fuse_bn(
    conv: nn.Conv1d,
    bn: nn.BatchNorm1d,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(weight, bias)`` of an equivalent conv with its BN fused in.

    Derivation:
        y = BN(conv(x)) = gamma * (conv(x) - mean) / std + beta
          = (gamma/std) * conv(x)  +  (beta - mean*gamma/std)
          = conv_fused(x)  where  w_fused = w * (gamma/std)
                                  b_fused = beta - mean * gamma / std
    """
    gamma = bn.weight
    beta  = bn.bias
    mean  = bn.running_mean
    var   = bn.running_var
    eps   = bn.eps
    std   = (var + eps).sqrt()
    # t: (out_channels, 1, 1) — broadcast over spatial and in_channel dims
    t = (gamma / std).reshape(-1, 1, 1)
    return conv.weight * t, beta - mean * gamma / std


# ---------------------------------------------------------------------------
# Structural-reparameterization depthwise conv
# ---------------------------------------------------------------------------

class _ReparamDWConv(nn.Module):
    """Structural-reparameterization depthwise Conv1d.

    **Training**: two parallel depthwise paths (large kernel + small kernel),
    each followed by BatchNorm1d.  Outputs are **added** before returning.

    **Inference**: after calling :meth:`merge_kernel`, both paths are fused
    into a single Conv1d with bias, eliminating the extra computation.

    Inspired by RepVGG / RepLKNet (as in the original ModernTCN reference).

    Args:
        channels:     Number of input/output channels (= M × D in the block).
        large_kernel: Large depthwise kernel size (must be odd, e.g. 51).
        small_kernel: Small depthwise kernel size (must be odd and ≤ large_kernel,
                      e.g. 5).  Zero-padded to ``large_kernel`` size before
                      merging.
    """

    def __init__(self, channels: int, large_kernel: int, small_kernel: int) -> None:
        super().__init__()
        assert small_kernel <= large_kernel, (
            f"small_kernel ({small_kernel}) must be ≤ large_kernel ({large_kernel})"
        )
        self.large_kernel = large_kernel
        self.small_kernel = small_kernel

        # Large-kernel path (+ BN)
        self.large_branch = nn.Sequential(
            nn.Conv1d(
                channels, channels,
                kernel_size=large_kernel,
                padding=large_kernel // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )

        # Small-kernel path (+ BN)
        self.small_branch = nn.Sequential(
            nn.Conv1d(
                channels, channels,
                kernel_size=small_kernel,
                padding=small_kernel // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, M*D, N)`` → ``(B, M*D, N)``."""
        if hasattr(self, "merged_conv"):
            return self.merged_conv(x)
        return self.large_branch(x) + self.small_branch(x)

    def merge_kernel(self) -> None:
        """Fuse large + small paths into one Conv1d for efficient inference.

        After this call the branch sub-modules are deleted and replaced by a
        single ``self.merged_conv`` Conv1d with bias.
        """
        if hasattr(self, "merged_conv"):
            return  # already merged

        # Fuse each branch: BN → conv weights
        k_large, b_large = _fuse_bn(self.large_branch[0], self.large_branch[1])
        k_small, b_small = _fuse_bn(self.small_branch[0], self.small_branch[1])

        # Zero-pad small kernel to match large kernel size
        pad = (self.large_kernel - self.small_kernel) // 2
        k_small_padded = F.pad(k_small, [pad, pad])

        k_merged = k_large + k_small_padded
        b_merged = b_large + b_small

        channels = self.large_branch[0].in_channels
        merged = nn.Conv1d(
            channels, channels,
            kernel_size=self.large_kernel,
            padding=self.large_kernel // 2,
            groups=channels,
            bias=True,
        )
        merged.weight.data = k_merged
        merged.bias.data   = b_merged
        self.merged_conv = merged

        # Remove training-time branches to free memory
        del self.large_branch, self.small_branch


# ---------------------------------------------------------------------------
# ModernTCN Block
# ---------------------------------------------------------------------------

class ModernTCNBlock(nn.Module):
    """One ModernTCN encoder block.

    Following the reference exactly, a **single** residual wraps the entire
    block (no sub-layer residuals as in Transformers):

    .. code-block:: text

        input ──────────────────────────────────────────────────► (+) ► output
                │                                                   │
                ▼                                                   │
          DWConv (structural reparam, groups=M*D)                   │
                │                                                   │
                ▼                                                   │
          BatchNorm1d (over B*M, D, N)                              │
                │                                                   │
                ▼                                                   │
          ConvFFN-1  (per-variate feature mixing, groups=M)         │
                │                                                   │
                ▼                                                   │
          ConvFFN-2  (cross-variate per-feature mixing, groups=D) ──┘

    Tensor shapes inside the block:

    * ``(B, M, D, N)`` — canonical format
    * Reshaped to ``(B, M*D, N)`` for DWConv and ConvFFN-1
    * Permuted to ``(B, D, M, N)`` → ``(B, D*M, N)`` for ConvFFN-2

    Args:
        d_model:      Embedding dimension D.
        nvars:        Number of variates M (= enc_in).
        large_kernel: Large DWConv kernel size.
        small_kernel: Small DWConv kernel size (structural reparam).
        ffn_ratio:    ConvFFN expansion ratio.  Hidden dim = d_model × ffn_ratio.
        dropout:      Dropout applied after each ConvFFN pointwise conv.
    """

    def __init__(
        self,
        d_model: int,
        nvars: int,
        large_kernel: int,
        small_kernel: int,
        ffn_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()

        d_ff = d_model * ffn_ratio

        # --- DWConv (structural reparam) ---
        self.dwconv = _ReparamDWConv(nvars * d_model, large_kernel, small_kernel)

        # --- BatchNorm after DWConv ---
        # Applied on (B*M, D, N): normalises each feature-dim across batch,
        # time, and variate — key design choice over LayerNorm (reference §4.2)
        self.norm = nn.BatchNorm1d(d_model)

        # --- ConvFFN-1: per-variate feature mixing (groups = M) ---
        # Each variate independently: D → d_ff → D
        self.ffn1_pw1   = nn.Conv1d(nvars * d_model, nvars * d_ff,
                                    kernel_size=1, groups=nvars, bias=False)
        self.ffn1_act   = nn.GELU()
        self.ffn1_pw2   = nn.Conv1d(nvars * d_ff,   nvars * d_model,
                                    kernel_size=1, groups=nvars, bias=False)
        self.ffn1_drop1 = nn.Dropout(dropout)
        self.ffn1_drop2 = nn.Dropout(dropout)

        # --- ConvFFN-2: cross-variate mixing per feature (groups = D) ---
        # For each of D feature dims, mix across M variates: M → d_ff → M
        self.ffn2_pw1   = nn.Conv1d(nvars * d_model, nvars * d_ff,
                                    kernel_size=1, groups=d_model, bias=False)
        self.ffn2_act   = nn.GELU()
        self.ffn2_pw2   = nn.Conv1d(nvars * d_ff,   nvars * d_model,
                                    kernel_size=1, groups=d_model, bias=False)
        self.ffn2_drop1 = nn.Dropout(dropout)
        self.ffn2_drop2 = nn.Dropout(dropout)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, M, D, N)``

        Returns:
            ``(B, M, D, N)``
        """
        B, M, D, N = x.shape
        residual = x

        # ── DWConv + BN ──────────────────────────────────────────────
        x = x.reshape(B, M * D, N)
        x = self.dwconv(x)                       # (B, M*D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)                         # BatchNorm1d over (B*M, D, N)
        x = x.reshape(B, M, D, N)

        # ── ConvFFN-1: per-variate feature mixing ─────────────────────
        x = x.reshape(B, M * D, N)
        x = self.ffn1_drop1(self.ffn1_pw1(x))   # (B, M*d_ff, N)
        x = self.ffn1_act(x)
        x = self.ffn1_drop2(self.ffn1_pw2(x))   # (B, M*D, N)
        x = x.reshape(B, M, D, N)

        # ── ConvFFN-2: cross-variate mixing (per feature dim) ─────────
        x = x.permute(0, 2, 1, 3)               # (B, D, M, N)
        x = x.reshape(B, D * M, N)
        x = self.ffn2_drop1(self.ffn2_pw1(x))   # (B, D*d_ff, N)
        x = self.ffn2_act(x)
        x = self.ffn2_drop2(self.ffn2_pw2(x))   # (B, D*M, N)
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)               # (B, M, D, N)

        # ── Single residual over entire block ─────────────────────────
        return residual + x
