#!/usr/bin/env bash
# dry_run.sh — sanity check before submitting the full sweep.
#
# Runs ONE config on ONE cell with one warmup batch, inside an interactive
# Slurm allocation.  Confirms that:
#   - the converted checkpoint loads,
#   - the tokenizer is found,
#   - the dataset directory is found,
#   - one batch produces a non-zero throughput record,
#   - peak GPU memory fits in 4× 40 GB.
#
# Usage (first grab a GPU node):
#   salloc --nodes=1 --gres=gpu:a100:4 --time=00:15:00 --partition=gpu
#   bash scripts/hpc/dry_run.sh
#
# Override defaults via env vars:
#   CONFIG=b CELL=pad50_cov20 bash scripts/hpc/dry_run.sh

set -euo pipefail

CONFIG="${CONFIG:-a}"
CELL="${CELL:-pad20_cov05}"

case "${CONFIG}" in
    a) CKPT_TAG="tp4"     ;;
    b) CKPT_TAG="tp2pp2"  ;;
    c) CKPT_TAG="tp1pp4"  ;;
    *) echo "Invalid CONFIG: ${CONFIG}"; exit 1 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/llm-inference-padding-study}"
MEGATRON_DIR="${MEGATRON_DIR:-$HOME/Megatron-LM}"
CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/qwen3-checkpoints}"
HF_QWEN3_DIR="${HF_QWEN3_DIR:-$SCRATCH/qwen3-32b-hf}"

export CHECKPOINT="${CKPT_ROOT}/qwen3-32b-mcore-${CKPT_TAG}"
export TOKENIZER_MODEL="${HF_QWEN3_DIR}"
export BENCHMARK_DATASET_DIR="${PROJECT_ROOT}/benchmark_dataset"
export RESULTS_DIR="${PROJECT_ROOT}/results_dryrun"
export PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}"

echo "Dry run:"
echo "  CONFIG       = ${CONFIG}"
echo "  CELL         = ${CELL}"
echo "  CHECKPOINT   = ${CHECKPOINT}"
echo "  RESULTS_DIR  = ${RESULTS_DIR}"
echo ""

cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_DIR}"

# The launch scripts hard-code --benchmark-cell-less invocation and
# --n-warmup 2.  For a single-cell dry run, we generate a one-shot copy of
# the launch script with --benchmark-cell and a smaller --n-warmup injected,
# then run that.
TMP_LAUNCH="$(mktemp /tmp/dryrun_launch.XXXXXX.sh)"
trap 'rm -f "${TMP_LAUNCH}"' EXIT

sed -E "
    s|--config-name config_${CONFIG} \\\\|--config-name config_${CONFIG}_dryrun --benchmark-cell ${CELL} \\\\|
    s|--n-warmup 2 \\\\|--n-warmup 1 \\\\|
" "scripts/launch_config_${CONFIG}.sh" > "${TMP_LAUNCH}"
chmod +x "${TMP_LAUNCH}"

echo "Running dry-run launch script …"
bash "${TMP_LAUNCH}"

echo ""
echo "Dry run complete.  Output files:"
ls -la "${RESULTS_DIR}/"
