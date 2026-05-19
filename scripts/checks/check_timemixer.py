"""Quick forward-pass sanity check for TimeMixer.

Run with:
    uv run check-timemixer
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


def main() -> None:
    print("=" * 55)
    print("  TimeMixer Forward-Pass Sanity Check")
    print("=" * 55)

    configs = SimpleNamespace(
        seq_len=96,
        pred_len=96,
        label_len=48,
        enc_in=7,
        c_out=7,
        d_model=16,
        d_ff=32,
        e_layers=2,
        dropout=0.1,
        down_sampling_layers=3,
        down_sampling_window=2,
        moving_avg=25,
    )

    from models.TimeMixer import TimeMixer

    model = TimeMixer(configs)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")

    B, T, C = 4, configs.seq_len, configs.enc_in
    x_enc      = torch.randn(B, T, C)
    x_mark_enc = torch.randn(B, T, 4)

    with torch.no_grad():
        out = model(x_enc, x_mark_enc)

    expected = (B, configs.pred_len, configs.c_out)
    assert out.shape == expected, (
        f"Shape mismatch — expected {expected}, got {tuple(out.shape)}"
    )
    assert out.isfinite().all(), "Output contains non-finite (NaN/Inf) values!"

    print(f"  Output shape : {tuple(out.shape)}  (expected {expected})")
    print(f"  All values finite : ✓")
    print("=" * 55)
    print("  ✓ TimeMixer forward pass OK.")
    print("=" * 55)


if __name__ == "__main__":
    main()
