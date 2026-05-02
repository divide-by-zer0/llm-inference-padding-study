#!/usr/bin/env bash
# convert_submit.slurm — Slurm wrapper for convert_checkpoints.sh.
#
# Submits the HF→Megatron checkpoint conversion as a batch job, so you don't
# have to babysit an `salloc` for ~75 min while three shardings are produced.
#
# Usage:
#   sbatch scripts/hpc/convert_submit.slurm
#
# Tail progress:
#   tail -f logs/qwen-convert-<jobid>.out
#
#SBATCH --job-name=qwen-convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Modules + venv  (same pattern as zaratan_submit.slurm)
# ---------------------------------------------------------------------------
PY_MODULE="${PY_MODULE:-python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2}"
VENV_DIR="${VENV_DIR:-$HOME/padding-bench-venv}"

module purge
module load "${PY_MODULE}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
unset PYTHONPATH

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
export PROJECT_BASE="${PROJECT_BASE:-$HOME/scratch.cmsc828}"
export PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_BASE/llm-inference-padding-study}"
export MEGATRON_DIR="${MEGATRON_DIR:-$PROJECT_BASE/Megatron-LM}"
export HF_QWEN_DIR="${HF_QWEN_DIR:-$PROJECT_BASE/qwen2.5-32b-hf}"
export CKPT_ROOT="${CKPT_ROOT:-$PROJECT_BASE/qwen-checkpoints}"

# ---------------------------------------------------------------------------
# Sanity checks before the GPU clock starts ticking
# ---------------------------------------------------------------------------
[[ -d "${PROJECT_ROOT}" ]]   || { echo "PROJECT_ROOT missing: ${PROJECT_ROOT}"; exit 1; }
[[ -d "${MEGATRON_DIR}" ]]   || { echo "MEGATRON_DIR missing: ${MEGATRON_DIR}"; exit 1; }
[[ -d "${HF_QWEN_DIR}" ]]    || { echo "HF_QWEN_DIR missing: ${HF_QWEN_DIR}"; exit 1; }

mkdir -p "${CKPT_ROOT}"

echo "==========================================================="
echo "  Job:        ${SLURM_JOB_NAME:-qwen-convert} (${SLURM_JOB_ID:-?})"
echo "  Started:    $(date)"
echo "  Megatron:   ${MEGATRON_DIR}"
echo "  HF model:   ${HF_QWEN_DIR}"
echo "  Output:     ${CKPT_ROOT}"
echo "==========================================================="

cd "${PROJECT_ROOT}"
bash scripts/hpc/convert_checkpoints.sh

echo ""
echo "==========================================================="
echo "  Finished:   $(date)"
echo "==========================================================="
