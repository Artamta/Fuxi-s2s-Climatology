#!/bin/bash
#SBATCH --job-name=fuxi_smoke
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/raj.ayush/s2s/fuxi_s2s_Hindcast/logs/fuxi_smoke_%j.out
#SBATCH --error=/home/raj.ayush/s2s/fuxi_s2s_Hindcast/logs/fuxi_smoke_%j.err

set -euo pipefail

REPO="/home/raj.ayush/s2s/fuxi_s2s_Hindcast"
FUXI_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/FuXi-S2S"
FUXI_PY="/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python"
CHECK_PY="${REPO}/scripts/check_smoke_output.py"

DATE="${FUXI_SMOKE_DATE:-20190617}"
MEMBERS="${FUXI_SMOKE_MEMBERS:-1}"
STEPS="${FUXI_SMOKE_STEPS:-2}"
SEED="${FUXI_SMOKE_SEED:-42}"
INPUT_FILE="${REPO}/data/ic_inputs/${DATE}/input.nc"
RAW_DIR="${REPO}/outputs/smoke/${DATE}/raw/${DATE}"

mkdir -p "${REPO}/logs" "${RAW_DIR}"

export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/fuxi_s2s/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "FuXi-S2S smoke test"
echo "Job ID    : ${SLURM_JOB_ID:-manual}"
echo "Node      : $(hostname)"
echo "Date      : ${DATE}"
echo "Input     : ${INPUT_FILE}"
echo "Members   : ${MEMBERS}"
echo "Steps     : ${STEPS}"
echo "Raw dir   : ${RAW_DIR}"
echo "Start     : $(date)"
echo "============================================================"

if [[ ! -s "${INPUT_FILE}" ]]; then
  echo "Missing input: ${INPUT_FILE}" >&2
  exit 2
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
python "${CHECK_PY}" \
  --raw-dir "${RAW_DIR}" \
  --members "${MEMBERS}" \
  --steps "${STEPS}"

echo "Done: $(date)"

