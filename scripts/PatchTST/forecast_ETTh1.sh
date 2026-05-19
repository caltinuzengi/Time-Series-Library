#!/usr/bin/env bash
# PatchTST — ETTh1 forecasting  (pred_len=96)
# Target MSE: ~0.370-0.390  (Nie et al. ICLR 2023, seq_len=96)
# Note: paper reports ~0.370 with seq_len=336; using seq_len=96 for
#       consistent internal comparison with TimesNet & TimeMixer.
set -euo pipefail

uv run python -u run.py \
  --model      PatchTST \
  --data       ETTh1 \
  --task       forecasting \
  --seq_len    96 \
  --label_len  48 \
  --pred_len   96 \
  --enc_in     7 \
  --c_out      7 \
  --d_model    128 \
  --d_ff       256 \
  --e_layers   3 \
  --n_heads    16 \
  --patch_len  16 \
  --stride     8 \
  --dropout    0.2 \
  --train_epochs  100 \
  --batch_size    64 \
  --learning_rate 1e-4 \
  --patience      10 \
  --num_workers   2 \
  --use_gpu \
  --seed       42
