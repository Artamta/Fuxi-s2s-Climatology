#!/bin/bash
# Run FuXi-S2S forecasts for the configured June 17 hindcast dates.
#
# Default is the control-only climatology run:
#   all 20 dates x 42 lead steps x 1 member
#
# For the full ensemble, submit with:
#   FUXI_MEMBERS=50 sbatch slurm/run_june17_forecasts.sbatch
#
# Forecast NetCDF outputs are written to /storage by default because /home is
# too small for the full hindcast archive.

#SBATCH --job-name=fuxi_june17
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1-20%1
#SBATCH --output=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/logs/fuxi_june17_%A_%a.out
#SBATCH --error=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/logs/fuxi_june17_%A_%a.err

set -euo pipefail

REPO="/home/raj.ayush/s2s/fuxi_s2s_Hindcast"
FUXI_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/FuXi-S2S"
FUXI_PY="/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python"
CHECK_PY="${FUXI_CHECK_PY:-python}"

DATES_FILE="${FUXI_DATES_FILE:-${REPO}/config/june17_dates_2002_2021.txt}"
INPUT_ROOT="${FUXI_INPUT_ROOT:-${REPO}/data/ic_inputs}"
OUTPUT_ROOT="${FUXI_OUTPUT_ROOT:-/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
DATE="$(sed -n "${TASK_ID}p" "${DATES_FILE}" | tr -d '[:space:]')"
MEMBERS="${FUXI_MEMBERS:-1}"
STEPS="${FUXI_STEPS:-42}"
SEED="${FUXI_SEED:-42}"
OVERWRITE="${FUXI_OVERWRITE:-0}"

INPUT_FILE="${INPUT_ROOT}/${DATE}/input.nc"
RAW_DIR="${OUTPUT_ROOT}/raw/${DATE}"

mkdir -p "${OUTPUT_ROOT}/logs" "${RAW_DIR%/*}"

export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/fuxi_s2s/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -z "${DATE}" ]]; then
  echo "No date found for task ${TASK_ID} in ${DATES_FILE}" >&2
  exit 2
fi

if [[ ! -s "${INPUT_FILE}" ]]; then
  echo "Missing input: ${INPUT_FILE}" >&2
  exit 2
fi

echo "============================================================"
echo "FuXi-S2S June 17 hindcast"
echo "Job ID    : ${SLURM_JOB_ID:-manual}"
echo "Task ID   : ${TASK_ID}"
echo "Node      : $(hostname)"
echo "Date      : ${DATE}"
echo "Input     : ${INPUT_FILE}"
echo "Members   : ${MEMBERS}"
echo "Steps     : ${STEPS}"
echo "Raw dir   : ${RAW_DIR}"
echo "CUDA devs : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start     : $(date)"
echo "============================================================"

if [[ -d "${RAW_DIR}" && "${OVERWRITE}" != "1" ]]; then
  cd "${REPO}"
  if "${CHECK_PY}" scripts/check_smoke_output.py \
      --raw-dir "${RAW_DIR}" \
      --members "${MEMBERS}" \
      --steps "${STEPS}"; then
    echo "Existing complete output found; skipping ${DATE}."
    exit 0
  fi
  echo "Existing output is incomplete; replacing ${RAW_DIR}."
fi

rm -rf "${RAW_DIR}"
mkdir -p "${RAW_DIR}"

cd "${FUXI_DIR}"
"${FUXI_PY}" inference_ensemble.py \
  --model "${FUXI_DIR}/model/fuxi_s2s.onnx" \
  --input "${INPUT_FILE}" \
  --device cuda \
  --total_step "${STEPS}" \
  --total_member "${MEMBERS}" \
  --seed "${SEED}" \
  --save_dir "${RAW_DIR}"

cd "${REPO}"
"${CHECK_PY}" scripts/check_smoke_output.py \
  --raw-dir "${RAW_DIR}" \
  --members "${MEMBERS}" \
  --steps "${STEPS}"

echo "Done: $(date)"
