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
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
# Logs go to $HOME (NOT scratch) because the project's scratch quota is full.
# /home/cadwani is hard-coded because SBATCH directives don't expand env vars.
#SBATCH --output=/home/cadwani/padding-results/logs/%x-%j.out
#SBATCH --error=/home/cadwani/padding-results/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cadwani@umd.edu

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
# Defaults match the Zaratan layout under ~/scratch.cmsc828.  Override via
# `--export` at sbatch time if your paths differ.
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/llm-inference-padding-study}"
export MEGATRON_DIR="${MEGATRON_DIR:-$HOME/scratch.cmsc828/Megatron-LM}"
export CKPT_ROOT="${CKPT_ROOT:-$HOME/scratch.cmsc828/qwen-checkpoints}"
export HF_QWEN_DIR="${HF_QWEN_DIR:-$HOME/scratch.cmsc828/qwen2.5-32b-hf}"

# Env vars consumed by launch_config_${CONFIG}.sh
export CHECKPOINT="${CKPT_ROOT}/qwen2.5-32b-mcore-${CKPT_TAG}"
export TOKENIZER_MODEL="${HF_QWEN_DIR}"
export BENCHMARK_DATASET_DIR="${PROJECT_ROOT}/benchmark_dataset"
# Results land in $HOME (separate quota from over-full project scratch).
export RESULTS_DIR="${RESULTS_DIR:-$HOME/padding-results}"

# ---------------------------------------------------------------------------
# Sanity checks before burning SUs
# ---------------------------------------------------------------------------
[[ -d "${CHECKPOINT}" ]]            || { echo "ERROR: checkpoint missing: ${CHECKPOINT}"; exit 1; }
[[ -d "${TOKENIZER_MODEL}" ]]       || { echo "ERROR: tokenizer dir missing: ${TOKENIZER_MODEL}"; exit 1; }
[[ -d "${BENCHMARK_DATASET_DIR}" ]] || { echo "ERROR: dataset dir missing: ${BENCHMARK_DATASET_DIR}"; exit 1; }
[[ -d "${MEGATRON_DIR}" ]]          || { echo "ERROR: Megatron-LM not found: ${MEGATRON_DIR}"; exit 1; }

mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Modules + venv
# ---------------------------------------------------------------------------
PY_MODULE="${PY_MODULE:-python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2}"
VENV_DIR="${VENV_DIR:-$HOME/padding-bench-venv}"

module purge
module load "${PY_MODULE}"
module load cuda/12.3.0/gcc/11.3.0/zen2
module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2
module load nccl/2.18.1-1/gcc/11.3.0/zen2
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
# Spack module sets PYTHONPATH; unset it so the venv's site-packages wins.
unset PYTHONPATH

# Make Megatron importable for run_padding_benchmark.py.
# (PYTHONPATH was just unset above; this becomes the only entry.)
export PYTHONPATH="${MEGATRON_DIR}"

# Transformer-engine 1.13 needs cuDNN 9.x at runtime, but Zaratan only ships
# cuDNN 8.x as a module.  PyTorch's wheel bundles libcudnn.so.9 inside the
# venv's nvidia/ directory; expose every nvidia/*/lib path so TE finds it.
export LD_LIBRARY_PATH=$(find "${VENV_DIR}/lib/python3.10/site-packages/nvidia" \
    -name "lib" -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}

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
