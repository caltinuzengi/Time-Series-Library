#!/usr/bin/env bash
# Full benchmark: 32 forecasting + 4 anomaly = 36 experiments
#
# Forecasting: 4 models × 4 ETT datasets × {24, 96} pred_len
# Anomaly:     4 models × SMD
set -euo pipefail

echo "========================================"
echo " FAZ 10 Benchmark — 36 experiments"
echo " Forecasting: 4 models × 4 datasets × 2 pred_lens = 32"
echo " Anomaly:     4 models × SMD              = 4"
echo "========================================"
echo ""

for MODEL in TimesNet TimeMixer PatchTST ModernTCN; do
  echo "--- $MODEL: Forecasting ---"
  bash "scripts/$MODEL/benchmark_forecasting.sh"
  echo ""

  echo "--- $MODEL: Anomaly Detection ---"
  bash "scripts/$MODEL/anomaly_SMD.sh"
  echo ""
done

echo "========================================"
echo " Benchmark complete: 36 experiments"
echo "========================================"
