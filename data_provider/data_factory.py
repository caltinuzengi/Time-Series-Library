"""
Data factory: maps dataset names and tasks to DataLoader instances.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from data_provider.data_loader import AnomalyDataset, ETTDataset

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Maps dataset name → (csv_filename, freq)
#   freq: "h" → hourly ETT borders, "t" → 15-minute ETT borders
DATASETS: dict[str, tuple[str, str]] = {
    "ETTh1": ("ETTh1.csv", "h"),
    "ETTh2": ("ETTh2.csv", "h"),
    "ETTm1": ("ETTm1.csv", "t"),
    "ETTm2": ("ETTm2.csv", "t"),
    "SMD":   ("SMD", None),   # directory-based, handled by AnomalyDataset
}


def get_dataloader(args, split: str) -> DataLoader:
    """Build and return a DataLoader for the requested dataset split.

    Selection logic:
    - ``args.task == "forecasting"``        → :class:`ETTDataset`
    - ``args.task == "anomaly_detection"``  → :class:`AnomalyDataset`

    Args:
        args:  Namespace / SimpleNamespace with fields:
               ``data``, ``task``, ``root_path``, ``target``,
               ``seq_len``, ``label_len``, ``pred_len``,
               ``batch_size``, ``num_workers`` (optional, default 4).
        split: One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.

    Raises:
        ValueError: If ``args.data`` or ``args.task`` is not recognised.
    """
    if args.data not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {args.data!r}. "
            f"Available: {list(DATASETS.keys())}"
        )

    data_file, freq = DATASETS[args.data]
    num_workers = getattr(args, "num_workers", 4)
    shuffle = split == "train"

    if args.task == "forecasting":
        dataset = ETTDataset(
            root_path=args.root_path,
            data_path=data_file,
            split=split,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            target=getattr(args, "target", "OT"),
            scale=True,
            freq=freq,
        )
    elif args.task == "anomaly_detection":
        # AnomalyDataset has no val split — val reuses train data
        actual_split = "train" if split == "val" else split
        dataset = AnomalyDataset(
            root_path=args.root_path,
            data_path=data_file,
            split=actual_split,
            win_size=args.seq_len,
            scale=True,
        )
    else:
        raise ValueError(
            f"Unknown task: {args.task!r}. "
            "Expected 'forecasting' or 'anomaly_detection'."
        )

    pin = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin,
    )
