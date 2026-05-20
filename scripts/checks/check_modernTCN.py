"""Quick forward-pass sanity check for ModernTCN.

Also verifies structural reparameterization:
  - Forward output must be identical before and after merge_reparams()

Run with:
    uv run check-moderntcn
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


def main() -> None:
    print("=" * 55)
    print("  ModernTCN Forward-Pass Sanity Check")
    print("=" * 55)

    configs = SimpleNamespace(
        seq_len=96,
        pred_len=96,
        enc_in=7,
        c_out=7,
        d_model=64,
        e_layers=2,
        patch_size=8,
        patch_stride=8,
        large_kernel=51,
        small_kernel=5,
        ffn_ratio=4,
        dropout=0.1,
    )

    from models.ModernTCN import ModernTCN

    model = ModernTCN(configs)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")
    print(f"  Num patches: {model.num_patches}  "
          f"(seq_len={configs.seq_len}, "
          f"patch_size={configs.patch_size}, patch_stride={configs.patch_stride})")

    B, T, C = 4, configs.seq_len, configs.enc_in
    x_enc      = torch.randn(B, T, C)
    x_mark_enc = torch.randn(B, T, 4)

    with torch.no_grad():
        out_pre = model(x_enc, x_mark_enc)

    expected = (B, configs.pred_len, configs.c_out)
    assert out_pre.shape == expected, (
        f"Shape mismatch — expected {expected}, got {tuple(out_pre.shape)}"
    )
    assert out_pre.isfinite().all(), "Output contains non-finite (NaN/Inf) values!"
    print(f"  Output shape : {tuple(out_pre.shape)}  (expected {expected})")
    print(f"  All values finite : ✓")

    # Verify structural reparameterization: output must be identical after merge
    model.merge_reparams()
    with torch.no_grad():
        out_post = model(x_enc, x_mark_enc)

    max_diff = (out_pre - out_post).abs().max().item()
    assert max_diff < 1e-4, (
        f"Reparam merge changed output by {max_diff:.2e} (expected < 1e-4)"
    )
    print(f"  Reparam merge max diff : {max_diff:.2e}  (< 1e-4) ✓")

    print("=" * 55)
    print("  ✓ ModernTCN forward pass OK.")
    print("=" * 55)


if __name__ == "__main__":
    main()
