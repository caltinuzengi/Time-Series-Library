#!/usr/bin/env bash
# PatchTST — Full benchmark: 4 ETT datasets × {24, 96} pred_len = 8 runs
set -euo pipefail

for DATA in ETTh1 ETTh2 ETTm1 ETTm2; do
  for PRED_LEN in 24 96; do
    echo ">>> PatchTST | $DATA | pred_len=$PRED_LEN"
    uv run python -u run.py \
      --model          PatchTST \
      --data           "$DATA" \
      --task           forecasting \
      --seq_len        96 \
      --label_len      48 \
      --pred_len       "$PRED_LEN" \
      --enc_in         7 \
      --c_out          7 \
      --d_model        128 \
      --d_ff           256 \
      --e_layers       3 \
      --n_heads        16 \
      --patch_len      16 \
      --stride         8 \
      --dropout        0.2 \
      --train_epochs   10 \
      --batch_size     32 \
      --learning_rate  1e-4 \
      --patience       10 \
      --num_workers    2 \
      --use_gpu \
      --seed           42
  done
done
