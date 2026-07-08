#!/bin/bash
#SBATCH --job-name=fuxi_ic_0617
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/fuxi_s2s_Hindcast/logs/fuxi_ic_0617_%j.out
#SBATCH --error=/home/raj.ayush/s2s/fuxi_s2s_Hindcast/logs/fuxi_ic_0617_%j.err

set -euo pipefail

REPO="/home/raj.ayush/s2s/fuxi_s2s_Hindcast"
PYTHON_BIN="${PYTHON_BIN:-/home/raj.ayush/.conda/envs/earth2/bin/python}"

cd "${REPO}"
mkdir -p logs data/ic_inputs

echo "Start: $(date)"
"${PYTHON_BIN}" scripts/make_june17_inputs.py \
  --years "${FUXI_YEARS:-2002:2021}" \
  --mmdd "${FUXI_MMDD:-0617}" \
  --output-dir "${FUXI_INPUT_DIR:-data/ic_inputs}" \
  --timeout "${FUXI_ARCO_TIMEOUT:-1800}"

python scripts/validate_fuxi_inputs.py \
  --years "${FUXI_YEARS:-2002:2021}" \
  --mmdd "${FUXI_MMDD:-0617}" \
  --input-dir "${FUXI_INPUT_DIR:-data/ic_inputs}"

echo "Done: $(date)"

