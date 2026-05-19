"""Quick forward-pass sanity check for PatchTST.

Run with:
    uv run check-patchtst
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


def main() -> None:
    print("=" * 55)
    print("  PatchTST Forward-Pass Sanity Check")
    print("=" * 55)

    configs = SimpleNamespace(
        seq_len=96,
        pred_len=96,
        enc_in=7,
        c_out=7,
        d_model=128,
        d_ff=256,
        e_layers=3,
        n_heads=16,
        patch_len=16,
        stride=8,
        dropout=0.2,
    )

    from models.PatchTST import PatchTST

    model = PatchTST(configs)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")

    from layers.PatchEmbedding import PatchEmbedding
    num_patches = model.patch_embed.num_patches
    print(f"  Num patches: {num_patches}  "
          f"(seq_len={configs.seq_len}, patch_len={configs.patch_len}, stride={configs.stride})")

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
    print("  ✓ PatchTST forward pass OK.")
    print("=" * 55)


if __name__ == "__main__":
    main()
