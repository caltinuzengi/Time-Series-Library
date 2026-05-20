"""Command-line entry point for training and evaluating time series models.

Usage examples:
  # TimesNet on ETTh1
  python run.py --model TimesNet --data ETTh1 --task forecasting \
    --seq_len 96 --pred_len 96 --label_len 48 \
    --enc_in 7 --c_out 7 --d_model 64 --d_ff 128 --e_layers 2 \
    --top_k 5 --num_kernels 6 --dropout 0.1 \
    --train_epochs 10 --batch_size 32 --learning_rate 1e-4 --use_gpu

  # Override pred_len without editing the script
  python run.py --model TimesNet --data ETTh1 --pred_len 336 ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from exp.exp_forecasting import ExpForecasting
from exp.exp_anomaly import ExpAnomaly


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Time Series Library — train / test CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- task & data ---
    p.add_argument("--model",  default="TimesNet", help="Model name (see MODEL_REGISTRY)")
    p.add_argument("--data",   default="ETTh1",    help="Dataset name")
    p.add_argument("--task",   default="forecasting",
                   choices=["forecasting", "anomaly_detection"])
    p.add_argument("--root_path", default="./data", help="Root directory for CSV files")

    # --- sequence lengths ---
    p.add_argument("--seq_len",   type=int, default=96,  help="Encoder input length")
    p.add_argument("--label_len", type=int, default=48,  help="Decoder overlap length")
    p.add_argument("--pred_len",  type=int, default=96,  help="Forecast horizon")

    # --- model architecture ---
    p.add_argument("--enc_in",      type=int,   default=7,    help="Encoder input features")
    p.add_argument("--c_out",       type=int,   default=7,    help="Output features")
    p.add_argument("--d_model",     type=int,   default=64,   help="Model dimension")
    p.add_argument("--d_ff",        type=int,   default=128,  help="FFN / InceptionBlock channels")
    p.add_argument("--e_layers",    type=int,   default=2,    help="Number of encoder layers")
    p.add_argument("--dropout",     type=float, default=0.1,  help="Dropout rate")

    # --- TimesNet-specific ---
    p.add_argument("--top_k",       type=int, default=5, help="FFT top-k periods")
    p.add_argument("--num_kernels", type=int, default=6, help="InceptionBlock kernels")

    # --- PatchTST-specific (Faz 5 — defaults set now for forward compatibility) ---
    p.add_argument("--n_heads",   type=int, default=8,  help="Attention heads (PatchTST)")
    p.add_argument("--patch_len", type=int, default=16, help="Patch length (PatchTST)")
    p.add_argument("--stride",    type=int, default=8,  help="Patch stride (PatchTST)")

    # --- ModernTCN-specific (Faz 6) ---
    p.add_argument("--patch_size",   type=int, default=8,  help="Conv stem patch size (ModernTCN)")
    p.add_argument("--patch_stride", type=int, default=8,  help="Conv stem stride (ModernTCN)")
    p.add_argument("--large_kernel", type=int, default=51, help="DWConv large kernel (ModernTCN)")
    p.add_argument("--small_kernel", type=int, default=5,  help="DWConv small kernel (ModernTCN)")
    p.add_argument("--ffn_ratio",    type=int, default=4,  help="ConvFFN expansion ratio (ModernTCN)")

    # --- TimeMixer-specific (Faz 4) ---
    p.add_argument("--down_sampling_layers", type=int, default=3,
                   help="Number of downsampling steps (TimeMixer)")
    p.add_argument("--down_sampling_window", type=int, default=2,
                   help="Downsampling factor between consecutive scales (TimeMixer)")
    p.add_argument("--moving_avg",           type=int, default=25,
                   help="Moving-average kernel size for series decomposition (TimeMixer)")

    # --- anomaly detection ---
    p.add_argument("--anomaly_ratio", type=float, default=1.0,
                   help="Anomaly percentage (used for threshold selection)")
    p.add_argument("--train_step", type=int, default=1,
                   help="Sliding-window stride during training (1=max overlap, "
                        "e.g. seq_len=non-overlapping).  Test always uses step=1.")

    # --- training ---
    p.add_argument("--train_epochs",   type=int,   default=10,   help="Max training epochs")
    p.add_argument("--batch_size",     type=int,   default=32,   help="Batch size")
    p.add_argument("--learning_rate",  type=float, default=1e-4, help="Initial learning rate")
    p.add_argument("--patience",       type=int,   default=3,    help="Early stopping patience")
    p.add_argument("--num_workers",    type=int,   default=2,    help="DataLoader workers (0=main process, Colab/WSL: ≤2)")

    # --- hardware ---
    p.add_argument("--use_gpu", action="store_true", default=False,
                   help="Use CUDA if available")
    p.add_argument("--gpu", type=int, default=0, help="GPU index")

    # --- reproducibility ---
    p.add_argument("--seed", type=int, default=42)

    # --- paths ---
    p.add_argument("--checkpoints", default="./checkpoints", help="Checkpoint root dir")
    p.add_argument("--results",     default="./results",     help="Results root dir")
    p.add_argument("--log_dir",     default="./logs",        help="Training log root dir")

    return p


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args) -> torch.device:
    if args.use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def build_checkpoint_path(args) -> str:
    tag = args.pred_len if args.task == "forecasting" else args.seq_len
    return os.path.join(
        args.checkpoints,
        f"{args.model}_{args.data}_{args.task}_{tag}",
    )


def save_results(args, metrics: dict, ts: str) -> None:
    """Persist metrics + config snapshot to results/."""
    tag = args.pred_len if args.task == "forecasting" else args.seq_len
    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_{args.data}_{args.task}_{tag}_{ts}.json"

    payload = {
        "metrics": metrics,
        "config": vars(args),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Results saved \u2192 {out_path}")


def save_epoch_log(args, epoch_logs: list, metrics: dict, ts: str) -> None:
    """Persist per-epoch training history + final metrics to logs/."""
    tag = args.pred_len if args.task == "forecasting" else args.seq_len
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_{args.data}_{args.task}_{tag}_{ts}.json"

    payload = {
        "config":      vars(args),
        "epoch_logs":  epoch_logs,
        "metrics":     metrics,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Epoch log saved \u2192 {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    args.device = resolve_device(args)
    args.checkpoint_path = build_checkpoint_path(args)

    ts = time.strftime("%Y%m%d_%H%M%S")   # shared timestamp for results + log files

    print(f"Device : {args.device}")
    print(f"Model  : {args.model}")
    print(f"Data   : {args.data}  seq={args.seq_len}  pred={args.pred_len}")

    if args.task == "forecasting":
        exp = ExpForecasting(args)
        exp.train()
        metrics = exp.test()
        save_results(args, metrics, ts)
        save_epoch_log(args, exp.epoch_logs, metrics, ts)
    elif args.task == "anomaly_detection":
        exp = ExpAnomaly(args)
        exp.train()
        metrics = exp.test()
        save_results(args, metrics, ts)
        save_epoch_log(args, exp.epoch_logs, metrics, ts)
    else:
        raise NotImplementedError(f"Task {args.task!r} is not implemented.")


if __name__ == "__main__":
    main()
