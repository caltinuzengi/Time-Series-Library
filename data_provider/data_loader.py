"""
Dataset classes for time series forecasting and anomaly detection.

Classes
-------
ETTDataset
    Electricity Transformer Temperature datasets (ETTh1/2, ETTm1/2).
    Returns 4-tuples: (seq_x, seq_y, seq_x_mark, seq_y_mark).

AnomalyDataset
    Server Machine Dataset (SMD) and compatible benchmarks.
    Sliding-window dataset for unsupervised anomaly detection.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


# ---------------------------------------------------------------------------
# ETT Forecasting Dataset
# ---------------------------------------------------------------------------

# Split boundaries (row indices) — ETT standard
_ETT_H_BORDERS = (12 * 30 * 24, 16 * 30 * 24)   # 8640, 11520  (hourly)
_ETT_M_BORDERS = (12 * 30 * 24 * 4, 16 * 30 * 24 * 4)  # 34560, 46080 (15-min)


class ETTDataset(Dataset):
    """ETT dataset with standard 60/20/20 train/val/test splits.

    The val and test splits overlap with the previous split by ``seq_len``
    rows to avoid information leakage at window boundaries.

    Args:
        root_path: Directory containing the CSV file.
        data_path: CSV filename (e.g. ``"ETTh1.csv"``).
        split:     One of ``"train"``, ``"val"``, ``"test"``.
        seq_len:   Look-back window length (L_x).
        label_len: Overlap between encoder and decoder inputs.
        pred_len:  Forecast horizon (L_y).
        target:    Name of the target column (e.g. ``"OT"``).
        scale:     If True, apply StandardScaler fitted on train split.
        freq:      ``"h"`` for hourly ETT, ``"t"`` for 15-minute ETT.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        split: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        target: str = "OT",
        scale: bool = True,
        freq: str = "h",
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target = target
        self.scale = scale
        self.freq = freq

        # ---- Load CSV -------------------------------------------------------
        csv_path = Path(root_path) / data_path
        if not csv_path.exists():
            raise FileNotFoundError(
                f"ETT data not found: {csv_path}\n"
                "Place the CSV file under data/ (see AGENT_PLAN.md §Görev 0.8)."
            )
        df = pd.read_csv(csv_path)

        # Determine split boundaries
        borders = _ETT_H_BORDERS if freq == "h" else _ETT_M_BORDERS
        train_end, val_end = borders

        if split == "train":
            start, end = 0, train_end
        elif split == "val":
            start, end = train_end - seq_len, val_end
        else:  # test
            start, end = val_end - seq_len, len(df)

        # ---- Features & target ----------------------------------------------
        # Move target column to last position, keep all others
        cols = [c for c in df.columns if c != "date" and c != target]
        cols.append(target)
        df_data = df[cols]

        # ---- Scaler (fit on train only) -------------------------------------
        self.scaler = StandardScaler()
        train_data = df_data.iloc[:train_end].values
        self.scaler.fit(train_data)

        data = (
            self.scaler.transform(df_data.values).astype(np.float32)
            if scale
            else df_data.values.astype(np.float32)
        )

        # ---- Time features --------------------------------------------------
        dates = pd.to_datetime(df["date"])
        data_stamp = time_features(pd.Series(dates))  # (total_len, 4)

        # ---- Slice to split --------------------------------------------------
        self.data_x = data[start:end]          # (split_len, C)
        self.data_y = data[start:end]
        self.data_stamp = data_stamp[start:end]

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        """Return a single sliding-window sample.

        Returns:
            seq_x:      (seq_len, C)              float32 — encoder input
            seq_y:      (label_len + pred_len, C) float32 — decoder input/target
            seq_x_mark: (seq_len, 4)              float32 — encoder time features
            seq_y_mark: (label_len + pred_len, 4) float32 — decoder time features
        """
        x_begin = idx
        x_end = x_begin + self.seq_len

        y_begin = x_end - self.label_len
        y_end = y_begin + self.label_len + self.pred_len

        seq_x = self.data_x[x_begin:x_end]
        seq_y = self.data_y[y_begin:y_end]
        seq_x_mark = self.data_stamp[x_begin:x_end]
        seq_y_mark = self.data_stamp[y_begin:y_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


# ---------------------------------------------------------------------------
# Anomaly Detection Dataset (SMD-style)
# ---------------------------------------------------------------------------


class AnomalyDataset(Dataset):
    """Sliding-window dataset for unsupervised anomaly detection.

    Compatible with the Server Machine Dataset (SMD) and similar benchmarks
    that store train/test splits as whitespace/comma-separated text files.

    Directory layout expected::

        root_path/SMD/
            train/<machine-id>.txt   (one file per entity)
            test/<machine-id>.txt
            test_label/<machine-id>.txt

    When a single concatenated file is provided (all entities merged), set
    ``data_path`` to that file directly.

    Args:
        root_path: Base data directory (e.g. ``"./data"``).
        data_path: Relative path to the dataset directory (e.g. ``"SMD"``).
        split:     ``"train"`` or ``"test"``.
        win_size:  Sliding-window length (= seq_len in config).
        scale:     If True, apply StandardScaler fitted on train split.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        split: str,
        win_size: int,
        scale: bool = True,
        step: int = 1,
    ) -> None:
        assert split in ("train", "test"), f"Unknown split: {split!r}"
        self.split = split
        self.win_size = win_size
        self.scale = scale
        self.step = max(1, step)

        base = Path(root_path) / data_path

        # Support both a directory-of-files layout and a single-file layout.
        def _load_txt(directory: Path) -> np.ndarray:
            if not directory.exists():
                raise FileNotFoundError(
                    f"SMD data not found: {directory}\n"
                    "Place SMD files under data/SMD/ (see AGENT_PLAN.md §Görev 0.8)."
                )
            if directory.is_file():
                return np.loadtxt(directory, delimiter=",", dtype=np.float32)
            # Concatenate all .txt files in the directory
            parts = sorted(directory.glob("*.txt"))
            if not parts:
                raise FileNotFoundError(f"No .txt files found in {directory}")
            return np.concatenate(
                [np.loadtxt(p, delimiter=",", dtype=np.float32) for p in parts],
                axis=0,
            )

        train_data = _load_txt(base / "train")
        test_data = _load_txt(base / "test")
        test_labels = _load_txt(base / "test_label")  # (N,) binary

        # Scaler: fit on train only
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)

        if scale:
            train_data = self.scaler.transform(train_data).astype(np.float32)
            test_data = self.scaler.transform(test_data).astype(np.float32)

        self.train = train_data
        self.test = test_data
        self.labels = test_labels.reshape(-1).astype(np.int64)

    def __len__(self) -> int:
        data = self.train if self.split == "train" else self.test
        n = max(0, len(data) - self.win_size + 1)
        return (n + self.step - 1) // self.step  # ceil division

    def __getitem__(self, idx: int):
        """Return a single sliding window.

        train split → ``(window,)``  shape ``(win_size, C)`` float32
        test  split → ``(window, label)`` shapes ``(win_size, C)`` and ``(win_size,)``
        """
        start = idx * self.step
        if self.split == "train":
            x = self.train[start : start + self.win_size]
            return x
        else:
            x = self.test[start : start + self.win_size]
            y = self.labels[start : start + self.win_size]
            return x, y
