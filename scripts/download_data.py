"""
Download benchmark datasets for Time-Series-Library.

Usage:
    python scripts/download_data.py           # downloads all ETT-small datasets
    python scripts/download_data.py --dataset ETTh1
    python scripts/download_data.py --dataset ETTh1 ETTh2
    python scripts/download_data.py --list    # show available datasets

Datasets
--------
ETTh1, ETTh2 : Hourly Electricity Transformer Temperature (17,420 rows × 8 cols)
ETTm1, ETTm2 : 15-minute Electricity Transformer Temperature (69,680 rows × 8 cols)

Official source: https://github.com/zhouhaoyi/ETDataset
Paper: Informer — AAAI 2021 Best Paper Award (Zhou et al.)
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# raw.githubusercontent.com URLs are the canonical way to download files
# directly from a public GitHub repository without the web UI overhead.
_BASE_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
)

DATASETS: dict[str, dict] = {
    "ETTh1": {
        "url": f"{_BASE_URL}/ETTh1.csv",
        "filename": "ETTh1.csv",
        "description": "Hourly ETT — station 1 (17,420 rows)",
    },
    "ETTh2": {
        "url": f"{_BASE_URL}/ETTh2.csv",
        "filename": "ETTh2.csv",
        "description": "Hourly ETT — station 2 (17,420 rows)",
    },
    "ETTm1": {
        "url": f"{_BASE_URL}/ETTm1.csv",
        "filename": "ETTm1.csv",
        "description": "15-min ETT — station 1 (69,680 rows)",
    },
    "ETTm2": {
        "url": f"{_BASE_URL}/ETTm2.csv",
        "filename": "ETTm2.csv",
        "description": "15-min ETT — station 2 (69,680 rows)",
    },
}

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _progress_hook(block_count: int, block_size: int, total_size: int) -> None:
    """Simple download progress indicator."""
    if total_size <= 0:
        print(f"\r  Downloaded {block_count * block_size / 1024:.0f} KB...", end="")
        return
    downloaded = block_count * block_size
    pct = min(100, downloaded * 100 // total_size)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%  ({downloaded / 1024:.0f} / {total_size / 1024:.0f} KB)", end="")


def download_dataset(name: str, dest_dir: Path, force: bool = False) -> bool:
    """Download a single dataset CSV to *dest_dir*.

    Args:
        name:     Dataset key (e.g. ``"ETTh1"``).
        dest_dir: Target directory — will be created if needed.
        force:    If True, re-download even if the file already exists.

    Returns:
        True if downloaded (or already present), False on error.
    """
    if name not in DATASETS:
        print(f"[ERROR] Unknown dataset: {name!r}. Use --list to see available datasets.")
        return False

    info = DATASETS[name]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / info["filename"]

    if dest_file.exists() and not force:
        size_kb = dest_file.stat().st_size / 1024
        print(f"  ✓ {name} already exists ({size_kb:.0f} KB) — skipping. Use --force to re-download.")
        return True

    print(f"  Downloading {name}: {info['description']}")
    print(f"  Source: {info['url']}")
    t0 = time.time()
    try:
        urllib.request.urlretrieve(info["url"], dest_file, reporthook=_progress_hook)
        elapsed = time.time() - t0
        size_kb = dest_file.stat().st_size / 1024
        print(f"\n  ✓ Saved → {dest_file}  ({size_kb:.0f} KB, {elapsed:.1f}s)")
        return True
    except Exception as exc:
        print(f"\n  ✗ Failed to download {name}: {exc}")
        if dest_file.exists():
            dest_file.unlink()  # remove partial download
        return False


def verify_dataset(name: str, dest_dir: Path) -> bool:
    """Basic integrity check: load CSV and verify shape."""
    try:
        import pandas as pd
    except ImportError:
        print("  [SKIP] pandas not available — skipping verification.")
        return True

    info = DATASETS[name]
    dest_file = dest_dir / info["filename"]
    if not dest_file.exists():
        return False

    df = pd.read_csv(dest_file)

    # Expected: columns = date + 6 load features + OT = 8 total
    expected_cols = 8
    if df.shape[1] != expected_cols:
        print(f"  ✗ {name}: expected {expected_cols} columns, got {df.shape[1]}")
        return False

    # Expected row counts (ETT standard)
    expected_rows = {"ETTh1": 17420, "ETTh2": 17420, "ETTm1": 69680, "ETTm2": 69680}
    if name in expected_rows and df.shape[0] != expected_rows[name]:
        print(
            f"  ✗ {name}: expected {expected_rows[name]} rows, got {df.shape[0]}"
        )
        return False

    print(f"  ✓ {name} verified: shape={df.shape}, columns={list(df.columns)}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ETT benchmark datasets from the official GitHub repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_data.py                        # all ETT datasets
  python scripts/download_data.py --dataset ETTh1 ETTh2  # specific datasets
  python scripts/download_data.py --force                # re-download even if exists
  python scripts/download_data.py --list                 # list available datasets
        """,
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        metavar="NAME",
        default=list(DATASETS.keys()),
        help="Dataset name(s) to download. Default: all ETT datasets.",
    )
    parser.add_argument(
        "--dest",
        default="./data",
        metavar="DIR",
        help="Destination directory. Default: ./data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-download integrity check.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name:<8}  {info['description']}")
            print(f"           {info['url']}")
        return

    dest_dir = Path(args.dest)
    print("=" * 60)
    print("ETT Dataset Downloader")
    print(f"Source : github.com/zhouhaoyi/ETDataset (Informer AAAI 2021)")
    print(f"Dest   : {dest_dir.resolve()}")
    print("=" * 60)

    # Validate requested names
    invalid = [n for n in args.dataset if n not in DATASETS]
    if invalid:
        print(f"[ERROR] Unknown dataset(s): {invalid}")
        print(f"Available: {list(DATASETS.keys())}")
        sys.exit(1)

    all_ok = True
    for name in args.dataset:
        print(f"\n[{name}]")
        ok = download_dataset(name, dest_dir, force=args.force)
        if ok and not args.no_verify:
            ok = verify_dataset(name, dest_dir)
        all_ok = all_ok and ok

    print("\n" + "=" * 60)
    if all_ok:
        print(f"✓ All downloads complete. Data ready in: {dest_dir.resolve()}")
        print("\nNext step:")
        print("  uv run check-data")
    else:
        print("✗ Some downloads failed — see above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
