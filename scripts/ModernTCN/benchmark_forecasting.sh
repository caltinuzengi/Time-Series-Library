#!/usr/bin/env bash
# ModernTCN — Full benchmark: 4 ETT datasets × {24, 96} pred_len = 8 runs
set -euo pipefail

for DATA in ETTh1 ETTh2 ETTm1 ETTm2; do
  for PRED_LEN in 24 96; do
    echo ">>> ModernTCN | $DATA | pred_len=$PRED_LEN"
    uv run python -u run.py \
      --model          ModernTCN \
      --data           "$DATA" \
      --task           forecasting \
      --seq_len        96 \
      --label_len      48 \
      --pred_len       "$PRED_LEN" \
      --enc_in         7 \
      --c_out          7 \
      --d_model        64 \
      --e_layers       2 \
      --patch_size     8 \
      --patch_stride   8 \
      --large_kernel   51 \
      --small_kernel   5 \
      --ffn_ratio      4 \
      --dropout        0.1 \
      --train_epochs   30 \
      --batch_size     32 \
      --learning_rate  1e-3 \
      --patience       5 \
      --num_workers    2 \
      --use_gpu \
      --seed           42
  done
done
