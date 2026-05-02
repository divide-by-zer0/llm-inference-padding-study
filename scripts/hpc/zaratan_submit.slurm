#!/usr/bin/env bash
# zaratan_submit.slurm — Slurm template for one parallelism config.
#
# Submitted via scripts/hpc/submit_all.sh, which exports CONFIG=a|b|c.
# Each invocation runs the full 16-cell × 5-batch sweep for one config.
#
#SBATCH --job-name=padding-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Required: CONFIG must be set by the caller (a, b, or c)
# ---------------------------------------------------------------------------
: "${CONFIG:?CONFIG env var must be set to a|b|c}"

case "${CONFIG}" in
    a) CKPT_TAG="tp4"     ;;
    b) CKPT_TAG="tp2pp2"  ;;
    c) CKPT_TAG="tp1pp4"  ;;
    *) echo "Invalid CONFIG: ${CONFIG}"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Paths — verify these match your Zaratan layout
# ---------------------------------------------------------------------------
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/llm-inference-padding-study}"
export MEGATRON_DIR="${MEGATRON_DIR:-$HOME/Megatron-LM}"
export CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/qwen-checkpoints}"
export HF_QWEN_DIR="${HF_QWEN_DIR:-$SCRATCH/qwen2.5-32b-hf}"

# Env vars consumed by launch_config_${CONFIG}.sh
export CHECKPOINT="${CKPT_ROOT}/qwen2.5-32b-mcore-${CKPT_TAG}"
export TOKENIZER_MODEL="${HF_QWEN_DIR}"
export BENCHMARK_DATASET_DIR="${PROJECT_ROOT}/benchmark_dataset"
export RESULTS_DIR="${PROJECT_ROOT}/results"

# ---------------------------------------------------------------------------
# Sanity checks before burning SUs
# ---------------------------------------------------------------------------
[[ -d "${CHECKPOINT}" ]]            || { echo "ERROR: checkpoint missing: ${CHECKPOINT}"; exit 1; }
[[ -d "${TOKENIZER_MODEL}" ]]       || { echo "ERROR: tokenizer dir missing: ${TOKENIZER_MODEL}"; exit 1; }
[[ -d "${BENCHMARK_DATASET_DIR}" ]] || { echo "ERROR: dataset dir missing: ${BENCHMARK_DATASET_DIR}"; exit 1; }
[[ -d "${MEGATRON_DIR}" ]]          || { echo "ERROR: Megatron-LM not found: ${MEGATRON_DIR}"; exit 1; }

mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Modules + conda env  (verify exact module names on Zaratan)
# ---------------------------------------------------------------------------
module purge
module load cuda/12.4
module load gcc
module load anaconda
# shellcheck disable=SC1091
source activate padding-bench

# Make Megatron importable for run_padding_benchmark.py
export PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}"

# Distributed defaults (NCCL on a single node)
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=8

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "==========================================================="
echo "  Job:          ${SLURM_JOB_NAME:-padding-bench} (${SLURM_JOB_ID:-?})"
echo "  Config:       ${CONFIG}  (${CKPT_TAG})"
echo "  Checkpoint:   ${CHECKPOINT}"
echo "  Dataset:      ${BENCHMARK_DATASET_DIR}"
echo "  Results:      ${RESULTS_DIR}"
echo "  Started:      $(date)"
echo "==========================================================="

cd "${PROJECT_ROOT}"
bash "scripts/launch_config_${CONFIG}.sh"

echo ""
echo "==========================================================="
echo "  Finished:     $(date)"
echo "==========================================================="
