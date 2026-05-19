#!/usr/bin/env bash
# TimeMixer — ETTh1 forecasting  (pred_len=96)
# Target MSE: ~0.360  (Wang et al. ICLR 2024, Table 1)
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
  --train_epochs   10 \
  --batch_size     16 \
  --learning_rate  0.01 \
  --patience       3 \
  --num_workers    2 \
  --use_gpu \
  --seed           42
