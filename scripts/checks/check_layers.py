"""Smoke tests for all Layer building blocks.

Checks:
  1. RevIN: round-trip (norm → denorm) preserves values.
  2. DataEmbedding: output shape correct.
  3. InceptionBlock: output shape correct.
  4. TimesBlock: output shape correct.

Run with:
  uv run check-layers
"""

from __future__ import annotations

import sys

import torch

PASS = "✓"
FAIL = "✗"


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status} {name}{suffix}")
    return ok


def check_revin() -> bool:
    from layers.RevIN import RevIN

    B, T, C = 4, 96, 7
    revin = RevIN(C, affine=True)
    x = torch.randn(B, T, C)
    x_norm = revin(x, "norm")
    x_back = revin(x_norm, "denorm")
    shape_ok = x_norm.shape == (B, T, C)
    roundtrip_ok = torch.allclose(x, x_back, atol=1e-4)
    return (
        _check("RevIN shape", shape_ok, str(tuple(x_norm.shape)))
        and _check("RevIN round-trip", roundtrip_ok)
    )


def check_embed() -> bool:
    from layers.Embed import DataEmbedding

    B, T, C, D = 4, 96, 7, 64
    emb = DataEmbedding(c_in=C, d_model=D, dropout=0.0)
    x = torch.randn(B, T, C)
    x_mark = torch.randn(B, T, 4)
    out = emb(x, x_mark)
    return _check("DataEmbedding shape", out.shape == (B, T, D), str(tuple(out.shape)))


def check_inception() -> bool:
    from layers.Conv_Blocks import InceptionBlock

    B, C, H, W = 4, 64, 12, 8
    block = InceptionBlock(in_channels=C, out_channels=C)
    x = torch.randn(B, C, H, W)
    out = block(x)
    return _check("InceptionBlock shape", out.shape == (B, C, H, W), str(tuple(out.shape)))


def check_timesblock() -> bool:
    from layers.Conv_Blocks import TimesBlock

    B, T, D = 4, 192, 64   # T = seq_len + pred_len
    block = TimesBlock(seq_len=96, pred_len=96, d_model=D, d_ff=D, top_k=5)
    x = torch.randn(B, T, D)
    out = block(x)
    return _check("TimesBlock shape", out.shape == (B, T, D), str(tuple(out.shape)))


def main() -> None:
    print("=" * 50)
    print("Layer Checks")
    print("=" * 50)

    results = [
        check_revin(),
        check_embed(),
        check_inception(),
        check_timesblock(),
    ]

    print("=" * 50)
    if all(results):
        print(f"{PASS} All layer checks passed.")
    else:
        n_fail = sum(not r for r in results)
        print(f"{FAIL} {n_fail} check(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
