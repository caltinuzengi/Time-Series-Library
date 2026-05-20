#!/usr/bin/env bash
# ModernTCN — Anomaly Detection on SMD (Server Machine Dataset)
#
# patch_size=8, patch_stride=8, seq_len=100 → num_patches=12
#
# Prerequisites:
#   uv run python scripts/download_anomaly_data.py
#
# Usage:
#   bash scripts/ModernTCN/anomaly_SMD.sh

set -euo pipefail

uv run python run.py \
  --model    ModernTCN \
  --task     anomaly_detection \
  --data     SMD \
  --root_path ./data \
  --seq_len  100 \
  --enc_in   38 \
  --c_out    38 \
  --d_model  64 \
  --d_ff     128 \
  --e_layers 2 \
  --patch_size   8 \
  --patch_stride 8 \
  --large_kernel 51 \
  --small_kernel 5 \
  --ffn_ratio    4 \
  --dropout  0.1 \
  --train_step    10 \
  --anomaly_ratio 0.5 \
  --train_epochs  1 \
  --batch_size    32 \
  --learning_rate 1e-4 \
  --patience      3 \
  --use_gpu
