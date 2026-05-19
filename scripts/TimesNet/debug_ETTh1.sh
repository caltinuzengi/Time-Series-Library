#!/usr/bin/env bash
# TimesNet — ETTh1 quick debug run (1 epoch, fast settings)
#
# Purpose: Verify the full pipeline works end-to-end:
#   data loading → training → validation → early stopping →
#   checkpointing → testing → result saving
#
# Expected runtime: ~1–2 min on Colab T4 / RTX 3050
# On CPU: ~5–10 min (omit --use_gpu)
set -euo pipefail

uv run python -u run.py \
  --model TimesNet \
  --data ETTh1 \
  --task forecasting \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --top_k 5 \
  --num_kernels 6 \
  --dropout 0.1 \
  --train_epochs 1 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --patience 1 \
  --num_workers 2 \
  --use_gpu \
  --seed 42
