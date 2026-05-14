"""Smoke test for the full TimesNet forward pass.

Checks:
  1. Output shape matches (B, pred_len, c_out).
  2. No NaN / Inf in output.
  3. Prints total parameter count.

Run with:
  uv run check-timesnet
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

PASS = "✓"
FAIL = "✗"


def main() -> None:
    from models.TimesNet import TimesNet

    cfg = SimpleNamespace(
        seq_len=96,
        pred_len=96,
        label_len=48,
        enc_in=7,
        c_out=7,
        d_model=64,
        d_ff=128,
        e_layers=2,
        top_k=5,
        num_kernels=6,
        dropout=0.1,
    )

    print("=" * 50)
    print("TimesNet Forward-Pass Check")
    print("=" * 50)

    model = TimesNet(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    B = 4
    x = torch.randn(B, cfg.seq_len, cfg.enc_in)
    x_mark = torch.randn(B, cfg.seq_len, 4)

    model.eval()
    with torch.no_grad():
        out = model(x, x_mark)

    expected = (B, cfg.pred_len, cfg.c_out)
    ok_shape = out.shape == expected
    ok_finite = torch.isfinite(out).all().item()

    print(f"  {'✓' if ok_shape else '✗'} Output shape: {tuple(out.shape)}  (expected {expected})")
    print(f"  {'✓' if ok_finite else '✗'} All values finite")

    print("=" * 50)
    if ok_shape and ok_finite:
        print(f"{PASS} TimesNet forward pass OK.")
    else:
        print(f"{FAIL} TimesNet check failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
