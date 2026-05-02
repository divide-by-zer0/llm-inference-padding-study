#!/usr/bin/env bash
# dry_run_submit.slurm — Slurm wrapper for dry_run.sh.
#
# Submits a 4× A100 single-cell sanity check as a batch job.  Use this when
# you don't want to babysit an `salloc` while the queue is slow.
#
# Usage:
#   sbatch scripts/hpc/dry_run_submit.slurm
#
# Override defaults via env vars:
#   CONFIG=b CELL=pad50_cov20 sbatch --export=ALL,CONFIG=b,CELL=pad50_cov20 \
#       scripts/hpc/dry_run_submit.slurm
#
# Tail the log:
#   tail -f logs/padding-dryrun-<jobid>.out
#
#SBATCH --job-name=padding-dryrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
# Logs go to $HOME (NOT scratch) because the project's scratch quota is full.
#SBATCH --output=/home/cadwani/padding-results/logs/%x-%j.out
#SBATCH --error=/home/cadwani/padding-results/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cadwani@umd.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — sane defaults; override at sbatch time via --export
# ---------------------------------------------------------------------------
CONFIG="${CONFIG:-a}"
CELL="${CELL:-pad20_cov05}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
export PROJECT_BASE="${PROJECT_BASE:-$HOME/scratch.cmsc828}"
export PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_BASE/llm-inference-padding-study}"
export MEGATRON_DIR="${MEGATRON_DIR:-$PROJECT_BASE/Megatron-LM}"
export HF_QWEN_DIR="${HF_QWEN_DIR:-$PROJECT_BASE/qwen2.5-32b-hf}"
export CKPT_ROOT="${CKPT_ROOT:-$PROJECT_BASE/qwen-checkpoints}"
export VENV_DIR="${VENV_DIR:-$HOME/padding-bench-venv}"

# ---------------------------------------------------------------------------
# Modules + venv
# ---------------------------------------------------------------------------
PY_MODULE="${PY_MODULE:-python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2}"

module purge
module load "${PY_MODULE}"
module load cuda/12.3.0/gcc/11.3.0/zen2
module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2
module load nccl/2.18.1-1/gcc/11.3.0/zen2

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
unset PYTHONPATH
export PYTHONPATH="${MEGATRON_DIR}"

# Bundled libcudnn.so.9 (and friends) for transformer-engine 1.13.
export LD_LIBRARY_PATH=$(find "${VENV_DIR}/lib/python3.10/site-packages/nvidia" \
    -name "lib" -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}

export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=8

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "==========================================================="
echo "  Dry run job:  ${SLURM_JOB_NAME:-padding-dryrun} (${SLURM_JOB_ID:-?})"
echo "  Node:         $(hostname)"
echo "  Config:       ${CONFIG}"
echo "  Cell:         ${CELL}"
echo "  Started:      $(date)"
echo "==========================================================="

cd "${PROJECT_ROOT}"
CONFIG="${CONFIG}" CELL="${CELL}" bash scripts/hpc/dry_run.sh

echo ""
echo "==========================================================="
echo "  Finished:     $(date)"
echo "==========================================================="
