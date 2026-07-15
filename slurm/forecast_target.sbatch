#!/bin/bash
#SBATCH --job-name=fuxi_jun17_wind_target
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/daily_mean_strict00z_june17_wind/logs/forecast_target_%j.out
#SBATCH --error=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/daily_mean_strict00z_june17_wind/logs/forecast_target_%j.err

set -euo pipefail

export LD_LIBRARY_PATH=/home/raj.ayush/.conda/envs/fuxi_s2s/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
ROOT=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/daily_mean_strict00z_june17_wind

exec "${PYTHON}" -u "${ROOT}/scripts/run_fuxi_wind_forecast.py" \
  --date 20260617 \
  --config "${ROOT}/config/run_config_target50.json"
