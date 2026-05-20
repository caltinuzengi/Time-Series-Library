#!/usr/bin/env bash
# TimesNet — Full benchmark: 4 ETT datasets × {24, 96} pred_len = 8 runs
set -euo pipefail

for DATA in ETTh1 ETTh2 ETTm1 ETTm2; do
  for PRED_LEN in 24 96; do
    echo ">>> TimesNet | $DATA | pred_len=$PRED_LEN"
    uv run python -u run.py \
      --model          TimesNet \
      --data           "$DATA" \
      --task           forecasting \
      --seq_len        96 \
      --label_len      48 \
      --pred_len       "$PRED_LEN" \
      --enc_in         7 \
      --c_out          7 \
      --d_model        64 \
      --d_ff           128 \
      --e_layers       2 \
      --top_k          5 \
      --num_kernels    6 \
      --dropout        0.1 \
      --train_epochs   10 \
      --batch_size     32 \
      --learning_rate  0.0001 \
      --patience       3 \
      --num_workers    2 \
      --use_gpu \
      --seed           42
  done
done
