#!/usr/bin/env bash
# convert_checkpoints.sh — convert HF Qwen3-32B safetensors to Megatron-Core
# format, sharded for each of the 3 parallelism configurations used in the
# benchmark.
#
# Run on a Zaratan compute node with ≥1 GPU (some converters need CUDA to
# initialise tensors).  Conversion is one-time; later inference jobs just
# reload the resulting checkpoints.
#
# Usage:
#   sbatch --time=02:00:00 --gres=gpu:a100:1 --mem=200G \
#       --wrap "bash scripts/hpc/convert_checkpoints.sh"
#
# Or run interactively after `salloc`:
#   bash scripts/hpc/convert_checkpoints.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# USER CONFIG — fill in for Zaratan
# ---------------------------------------------------------------------------
MEGATRON_DIR="${MEGATRON_DIR:-$HOME/Megatron-LM}"
HF_QWEN3_DIR="${HF_QWEN3_DIR:-$SCRATCH/qwen3-32b-hf}"      # downloaded HF weights
CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/qwen3-checkpoints}"       # output root

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
[[ -d "${MEGATRON_DIR}" ]]  || { echo "MEGATRON_DIR not found: ${MEGATRON_DIR}"; exit 1; }
[[ -d "${HF_QWEN3_DIR}" ]]  || { echo "HF_QWEN3_DIR not found: ${HF_QWEN3_DIR}"; exit 1; }
mkdir -p "${CKPT_ROOT}"

CONVERT_TOOL="${MEGATRON_DIR}/tools/checkpoint/convert.py"
[[ -f "${CONVERT_TOOL}" ]] || { echo "convert.py not found at ${CONVERT_TOOL}"; exit 1; }

# ---------------------------------------------------------------------------
# Conversion configurations
# ---------------------------------------------------------------------------
# Format: "tag:TP:PP"
CONFIGS=(
    "tp4:4:1"
    "tp2pp2:2:2"
    "tp1pp4:1:4"
)

for spec in "${CONFIGS[@]}"; do
    IFS=":" read -r TAG TP PP <<< "${spec}"
    OUT_DIR="${CKPT_ROOT}/qwen3-32b-mcore-${TAG}"

    if [[ -d "${OUT_DIR}" && -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]]; then
        echo "[convert] Skipping ${TAG}: ${OUT_DIR} already populated"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "[convert] Sharding to TP=${TP}, PP=${PP}  →  ${OUT_DIR}"
    echo "============================================================"

    # NOTE: --loader name depends on the Megatron-LM commit.  Recent versions
    # ship a Qwen-specific loader; older versions reuse `llama_mistral` for
    # decoder-only models with GQA + RMSNorm + SwiGLU.  Verify with:
    #   ls ${MEGATRON_DIR}/tools/checkpoint/loader_*.py
    # and adjust LOADER below if needed.
    LOADER="qwen"
    if [[ ! -f "${MEGATRON_DIR}/tools/checkpoint/loader_${LOADER}.py" ]]; then
        LOADER="llama_mistral"
        echo "[convert] WARNING: no loader_qwen.py; falling back to ${LOADER}"
    fi

    python "${CONVERT_TOOL}" \
        --model-type GPT \
        --loader "${LOADER}" \
        --saver  mcore \
        --target-tensor-parallel-size  "${TP}" \
        --target-pipeline-parallel-size "${PP}" \
        --load-dir "${HF_QWEN3_DIR}" \
        --save-dir "${OUT_DIR}" \
        --tokenizer-model "${HF_QWEN3_DIR}" \
        --bf16

    echo "[convert] ${TAG} done."
done

echo ""
echo "[convert] All conversions complete.  Sharded checkpoints under:"
ls -d "${CKPT_ROOT}"/qwen3-32b-mcore-* 2>/dev/null || true
