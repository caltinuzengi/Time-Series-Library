"""
Data pipeline sanity check — run with: uv run check-data

Verifies that ETTh1 dataloaders produce the expected batch shapes for all
three splits (train / val / test).

Expected output (seq_len=96, label_len=48, pred_len=96, batch_size=32):
    train | batch_x: (32, 96, 7)   batch_y: (32, 144, 7)
    val   | batch_x: (32, 96, 7)   batch_y: (32, 144, 7)
    test  | batch_x: (32, 96, 7)   batch_y: (32, 144, 7)
"""

from __future__ import annotations

import sys
from types import SimpleNamespace


def main() -> None:
    # Inline import so import errors are caught gracefully
    try:
        from data_provider.data_factory import get_dataloader
    except ImportError as exc:
        print(f"[ERROR] Cannot import data_provider: {exc}", file=sys.stderr)
        sys.exit(1)

    args = SimpleNamespace(
        data="ETTh1",
        task="forecasting",
        root_path="./data",
        target="OT",
        seq_len=96,
        label_len=48,
        pred_len=96,
        batch_size=32,
        num_workers=0,  # 0 for quick check (avoids multiprocessing overhead)
    )

    print("=" * 60)
    print("Data Pipeline Check — ETTh1 Forecasting")
    print("=" * 60)

    all_ok = True
    for split in ("train", "val", "test"):
        try:
            loader = get_dataloader(args, split)
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))

            expected_x = (args.batch_size, args.seq_len, 7)
            expected_y = (args.batch_size, args.label_len + args.pred_len, 7)
            expected_xm = (args.batch_size, args.seq_len, 4)
            expected_ym = (args.batch_size, args.label_len + args.pred_len, 4)

            x_ok = tuple(batch_x.shape) == expected_x
            y_ok = tuple(batch_y.shape) == expected_y
            xm_ok = tuple(batch_x_mark.shape) == expected_xm
            ym_ok = tuple(batch_y_mark.shape) == expected_ym

            status = "✓" if (x_ok and y_ok and xm_ok and ym_ok) else "✗"
            print(
                f"  {status} {split:5s} | "
                f"batch_x: {tuple(batch_x.shape)}  "
                f"batch_y: {tuple(batch_y.shape)}  "
                f"x_mark: {tuple(batch_x_mark.shape)}  "
                f"y_mark: {tuple(batch_y_mark.shape)}"
            )

            if not (x_ok and y_ok):
                all_ok = False
                if not x_ok:
                    print(f"         Expected batch_x: {expected_x}")
                if not y_ok:
                    print(f"         Expected batch_y: {expected_y}")

        except FileNotFoundError as exc:
            print(f"  ✗ {split:5s} | {exc}")
            print(
                "\n[INFO] ETTh1.csv not found. Download instructions:\n"
                "  1. Get ETTh1.csv from https://github.com/zhouhaoyi/ETDataset\n"
                "  2. Place it under ./data/ETTh1.csv\n"
                "  3. Re-run: uv run check-data\n"
                "(A download script will be added before Faz 1.)"
            )
            all_ok = False
            break

    print("=" * 60)
    if all_ok:
        print("✓ Data pipeline check passed.")
    else:
        print("✗ Data pipeline check FAILED — see above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
