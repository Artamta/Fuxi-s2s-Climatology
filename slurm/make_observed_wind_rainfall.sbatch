#!/usr/bin/env bash
#SBATCH --job-name=obs_wind_rain
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/era5_gt/logs/obs_wind_rain_%j.out
#SBATCH --error=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/era5_gt/logs/obs_wind_rain_%j.err

set -euo pipefail

REPO=/home/raj.ayush/s2s/fuxi_s2s_Hindcast
DOWNLOAD_PY=/home/raj.ayush/.conda/envs/fuxi/bin/python
PLOT_PY=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
JUNE_JOB_ID=${CDS_JUNE_JOB_ID:?export CDS_JUNE_JOB_ID with the accepted June CDS request ID}

mkdir -p /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/era5_gt/logs
mkdir -p /tmp/mpl-obs-wind-rain-${SLURM_JOB_ID} /tmp/xdg-obs-wind-rain-${SLURM_JOB_ID}

cd "${REPO}"
"${DOWNLOAD_PY}" scripts/download_era5_wind850_gt.py \
  --resume-june-job-id "${JUNE_JOB_ID}" \
  --overwrite

MPLCONFIGDIR=/tmp/mpl-obs-wind-rain-${SLURM_JOB_ID} \
XDG_CACHE_HOME=/tmp/xdg-obs-wind-rain-${SLURM_JOB_ID} \
"${PLOT_PY}" scripts/plot_observed_weekly_wind_rainfall.py
