"""Download the SMD (Server Machine Dataset) for anomaly detection.

Source:
    OmniAnomaly GitHub repository:
    https://github.com/NetManAIOps/OmniAnomaly

Expected layout after download:
    data/SMD/
        train/          # machine-x-y.txt — 38 comma-separated float columns
        test/           # same filenames as train/
        test_label/     # same filenames — one integer (0/1) per line

Usage:
    uv run python scripts/download_anomaly_data.py

The script tries to use sparse git checkout (no large repo history).
If git is unavailable, it prints manual instructions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SMD_URL  = "https://github.com/NetManAIOps/OmniAnomaly.git"
SMD_DIR  = "ServerMachineDataset"
DEST_DIR = Path("data/SMD")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def download_via_sparse_checkout() -> None:
    """Clone only the ServerMachineDataset/ folder via sparse checkout."""
    import tempfile

    print("Downloading SMD via git sparse-checkout …")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "omnianomaly"

        _run(["git", "clone", "--no-checkout", "--depth=1", SMD_URL, str(tmp_path)])
        _run(["git", "-C", str(tmp_path), "sparse-checkout", "init", "--cone"])
        _run(["git", "-C", str(tmp_path), "sparse-checkout", "set", SMD_DIR])
        _run(["git", "-C", str(tmp_path), "checkout"])

        src = tmp_path / SMD_DIR

        # Map OmniAnomaly sub-dirs → our expected layout
        mapping = {
            "train":      DEST_DIR / "train",
            "test":       DEST_DIR / "test",
            "test_label": DEST_DIR / "test_label",
        }

        for src_sub, dst_sub in mapping.items():
            dst_sub.mkdir(parents=True, exist_ok=True)
            for f in (src / src_sub).glob("*.txt"):
                import shutil
                shutil.copy(f, dst_sub / f.name)

    print(f"SMD downloaded → {DEST_DIR.resolve()}")
    _print_summary()


def _print_summary() -> None:
    for sub in ("train", "test", "test_label"):
        d = DEST_DIR / sub
        n = len(list(d.glob("*.txt"))) if d.exists() else 0
        print(f"  {sub:12s}  {n} files")


def _print_manual_instructions() -> None:
    print(
        "\nAutomatic download failed or git is not available.\n"
        "\nManual download steps:\n"
        "  1. Open: https://github.com/NetManAIOps/OmniAnomaly\n"
        "  2. Navigate to: ServerMachineDataset/\n"
        "  3. Download the three sub-directories:\n"
        "       train/       → data/SMD/train/\n"
        "       test/        → data/SMD/test/\n"
        "       test_label/  → data/SMD/test_label/\n"
        "\nFile format:\n"
        "  train/*.txt and test/*.txt — CSV, 38 float columns, one row per timestep\n"
        "  test_label/*.txt           — one integer (0 = normal, 1 = anomaly) per line\n"
    )


def main() -> None:
    if DEST_DIR.exists() and any(DEST_DIR.iterdir()):
        print(f"SMD directory already exists at {DEST_DIR.resolve()}")
        _print_summary()
        return

    # Try automatic download
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        download_via_sparse_checkout()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Automatic download failed: {e}")
        _print_manual_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()
