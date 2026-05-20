#!/usr/bin/env bash
# ModernTCN — ETTh1 forecasting  (pred_len=96)
# Target MSE: ~0.380-0.420  (Luo & Wang ICLR 2024, seq_len=96)
# Note: paper reports with seq_len=512; using seq_len=96 for consistent
#       internal comparison with TimesNet, TimeMixer & PatchTST.
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
  --train_epochs 30 \
  --batch_size   32 \
  --learning_rate 1e-3 \
  --patience     5 \
  --num_workers  2 \
  --use_gpu \
  --seed         42
