#!/usr/bin/env bash
# PatchTST — Anomaly Detection on SMD (Server Machine Dataset)
#
# Note: patch_len=16, stride=8 with seq_len=100 → num_patches = ⌈(100-16)/8⌉+1 = 12
#
# Prerequisites:
#   uv run python scripts/download_anomaly_data.py
#
# Usage:
#   bash scripts/PatchTST/anomaly_SMD.sh

set -euo pipefail

uv run python run.py \
  --model    PatchTST \
  --task     anomaly_detection \
  --data     SMD \
  --root_path ./data \
  --seq_len  100 \
  --enc_in   38 \
  --c_out    38 \
  --d_model  64 \
  --d_ff     128 \
  --e_layers 3 \
  --n_heads  8 \
  --patch_len 16 \
  --stride    8 \
  --dropout  0.1 \
  --train_step    10 \
  --anomaly_ratio 0.5 \
  --train_epochs  1 \
  --batch_size    32 \
  --learning_rate 1e-4 \
  --patience      3 \
  --use_gpu
