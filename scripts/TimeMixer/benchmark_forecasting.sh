#!/usr/bin/env bash
# TimeMixer — Full benchmark: 4 ETT datasets × {24, 96} pred_len = 8 runs
set -euo pipefail

for DATA in ETTh1 ETTh2 ETTm1 ETTm2; do
  for PRED_LEN in 24 96; do
    echo ">>> TimeMixer | $DATA | pred_len=$PRED_LEN"
    uv run python -u run.py \
      --model                TimeMixer \
      --data                 "$DATA" \
      --task                 forecasting \
      --seq_len              96 \
      --label_len            48 \
      --pred_len             "$PRED_LEN" \
      --enc_in               7 \
      --c_out                7 \
      --d_model              16 \
      --d_ff                 32 \
      --e_layers             2 \
      --down_sampling_layers 3 \
      --down_sampling_window 2 \
      --moving_avg           25 \
      --dropout              0.1 \
      --train_epochs         20 \
      --batch_size           16 \
      --learning_rate        0.01 \
      --patience             5 \
      --num_workers          2 \
      --use_gpu \
      --seed                 42
  done
done
