#!/usr/bin/env bash
# TimeMixer — ETTh1 quick 1-epoch debug run
# Use this to verify the pipeline before launching a full training run.
set -euo pipefail

uv run python -u run.py \
  --model          TimeMixer \
  --data           ETTh1 \
  --task           forecasting \
  --seq_len        96 \
  --label_len      48 \
  --pred_len       96 \
  --enc_in         7 \
  --c_out          7 \
  --d_model        16 \
  --d_ff           32 \
  --e_layers       2 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --moving_avg     25 \
  --dropout        0.1 \
  --train_epochs   1 \
  --batch_size     16 \
  --learning_rate  0.01 \
  --patience       3 \
  --num_workers    2 \
  --use_gpu \
  --seed           42
