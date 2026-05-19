#!/usr/bin/env bash
# TimesNet — ETTh1 forecasting, pred_len=96
# Reference: MSE ~ 0.384 (TSLib ICLR 2023)
set -euo pipefail

uv run python run.py \
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
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --patience 3 \
  --use_gpu \
  --seed 42
