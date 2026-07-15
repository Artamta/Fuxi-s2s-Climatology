#!/bin/bash
#SBATCH --job-name=fuxi_jun17_wind_plot
#SBATCH --partition=gpu_prio
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/daily_mean_strict00z_june17_wind/logs/postprocess_%j.out
#SBATCH --error=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/daily_mean_strict00z_june17_wind/logs/postprocess_%j.err

set -euo pipefail

PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
ROOT=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/daily_mean_strict00z_june17_wind
HOME_OUTPUT=/home/raj.ayush/s2s/fuxi_s2s_Hindcast/outputs/wind_rainfall_anomaly_20260617

export PROJ_DATA=/home/raj.ayush/.conda/envs/s2s-hind/share/proj
export PROJ_LIB=/home/raj.ayush/.conda/envs/s2s-hind/share/proj

"${PYTHON}" -u "${ROOT}/scripts/build_weekly_wind_analysis.py"
"${PYTHON}" -u "${ROOT}/scripts/plot_weekly_wind_rainfall.py"

mkdir -p "${HOME_OUTPUT}"
cp "${ROOT}"/figures/*.png "${HOME_OUTPUT}/"
cp "${ROOT}"/figures/*.pdf "${HOME_OUTPUT}/"
cp "${ROOT}"/figures/plot_manifest.json "${HOME_OUTPUT}/"
cp "${ROOT}"/analysis/fuxi_weekly_wind_rainfall_anomaly_20260617.nc "${HOME_OUTPUT}/"
cp "${ROOT}"/analysis/west_coast_weekly_diagnostics.csv "${HOME_OUTPUT}/"
cp "${ROOT}"/analysis/build_manifest.json "${HOME_OUTPUT}/analysis_build_manifest.json"

printf 'Copied June-17 wind products to %s\n' "${HOME_OUTPUT}"
