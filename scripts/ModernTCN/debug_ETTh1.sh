#!/usr/bin/env bash
# ModernTCN — ETTh1 quick 1-epoch debug run
# Use this to verify the pipeline before launching a full training run.
set -euo pipefail

uv run python -u run.py \
  --model        ModernTCN \
  --data         ETTh1 \
  --task         forecasting \
  --seq_len      96 \
  --label_len    48 \
  --pred_len     96 \
  --enc_in       7 \
  --c_out        7 \
  --d_model      64 \
  --e_layers     2 \
  --patch_size   8 \
  --patch_stride 8 \
  --large_kernel 51 \
  --small_kernel 5 \
  --ffn_ratio    4 \
  --dropout      0.1 \
  --train_epochs 1 \
  --batch_size   32 \
  --learning_rate 1e-3 \
  --patience     5 \
  --num_workers  2 \
  --use_gpu \
  --seed         42
