#!/usr/bin/env bash
# convert_checkpoints.sh — convert HF Qwen2.5-32B safetensors to Megatron-Core
# format, sharded for each of the 3 parallelism configurations used in the
# benchmark.
#
# Loader: tools/checkpoint/loader_llama_mistral.py with --model-size qwen2.5.
# This is the only HF→mcore loader in Megatron-LM main that supports Qwen
# (no Qwen3 loader exists; Qwen3 has q_norm/k_norm layers that this loader
#  does not handle, which is why we use Qwen2.5 instead).
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
HF_QWEN_DIR="${HF_QWEN_DIR:-$SCRATCH/qwen2.5-32b-hf}"      # downloaded HF weights
CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/qwen-checkpoints}"        # output root

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
[[ -d "${MEGATRON_DIR}" ]] || { echo "MEGATRON_DIR not found: ${MEGATRON_DIR}"; exit 1; }
[[ -d "${HF_QWEN_DIR}" ]]  || { echo "HF_QWEN_DIR not found: ${HF_QWEN_DIR}"; exit 1; }
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
    OUT_DIR="${CKPT_ROOT}/qwen2.5-32b-mcore-${TAG}"

    if [[ -d "${OUT_DIR}" && -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]]; then
        echo "[convert] Skipping ${TAG}: ${OUT_DIR} already populated"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "[convert] Sharding to TP=${TP}, PP=${PP}  →  ${OUT_DIR}"
    echo "============================================================"

    python "${CONVERT_TOOL}" \
        --model-type GPT \
        --loader llama_mistral \
        --saver  mcore \
        --model-size qwen2.5 \
        --checkpoint-type hf \
        --target-tensor-parallel-size  "${TP}" \
        --target-pipeline-parallel-size "${PP}" \
        --load-dir "${HF_QWEN_DIR}" \
        --save-dir "${OUT_DIR}" \
        --tokenizer-model "${HF_QWEN_DIR}" \
        --bf16

    echo "[convert] ${TAG} done."
done

echo ""
echo "[convert] All conversions complete.  Sharded checkpoints under:"
ls -d "${CKPT_ROOT}"/qwen2.5-32b-mcore-* 2>/dev/null || true
