#!/usr/bin/env bash
# convert_checkpoint_one.sh - shared helper for one HF -> Megatron-Core sharding.
#
# Prefer the config-specific wrappers:
#   bash scripts/hpc/convert_checkpoints_tp4.sh
#   bash scripts/hpc/convert_checkpoints_tp2pp2.sh
#   bash scripts/hpc/convert_checkpoints_tp1pp4.sh
#
# This one-checkpoint-at-a-time path is intended for tight scratch quotas where
# the HF safetensors plus all three converted checkpoints cannot coexist.

set -euo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

: "${CKPT_TAG:?CKPT_TAG must be set, e.g. tp4}"
: "${TARGET_TP:?TARGET_TP must be set}"
: "${TARGET_PP:?TARGET_PP must be set}"

MEGATRON_DIR="${MEGATRON_DIR:-$HOME/scratch.cmsc828/Megatron-LM}"
HF_QWEN_DIR="${HF_QWEN_DIR:-$HOME/scratch.cmsc828/qwen2.5-32b-hf}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/scratch.cmsc828/qwen-checkpoints}"
OUT_DIR="${OUT_DIR:-${CKPT_ROOT}/qwen2.5-32b-mcore-${CKPT_TAG}}"

[[ -d "${MEGATRON_DIR}" ]] || { echo "MEGATRON_DIR not found: ${MEGATRON_DIR}"; exit 1; }
[[ -d "${HF_QWEN_DIR}" ]]  || { echo "HF_QWEN_DIR not found: ${HF_QWEN_DIR}"; exit 1; }
mkdir -p "${CKPT_ROOT}"

CONVERT_TOOL="${MEGATRON_DIR}/tools/checkpoint/convert.py"
[[ -f "${CONVERT_TOOL}" ]] || { echo "convert.py not found at ${CONVERT_TOOL}"; exit 1; }

if [[ -f "${OUT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "[convert] ${OUT_DIR} already has latest_checkpointed_iteration.txt."
    echo "[convert] Delete or move it first if you want to rebuild this checkpoint."
    exit 0
fi

if [[ -d "${OUT_DIR}" && -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]]; then
    echo "[convert] ${OUT_DIR} exists but does not look complete."
    echo "[convert] Remove the partial checkpoint before retrying:"
    echo "          rm -rf \"${OUT_DIR}\""
    exit 1
fi

echo ""
echo "============================================================"
echo "[convert] Qwen2.5-32B -> Megatron-Core"
echo "[convert] Tag:       ${CKPT_TAG}"
echo "[convert] TP/PP:     ${TARGET_TP}/${TARGET_PP}"
echo "[convert] HF input:  ${HF_QWEN_DIR}"
echo "[convert] Output:    ${OUT_DIR}"
echo "============================================================"

python "${CONVERT_TOOL}" \
    --model-type GPT \
    --loader llama_mistral \
    --saver mcore \
    --model-size qwen2.5 \
    --checkpoint-type hf \
    --megatron-path "${MEGATRON_DIR}" \
    --target-tensor-parallel-size "${TARGET_TP}" \
    --target-pipeline-parallel-size "${TARGET_PP}" \
    --load-dir "${HF_QWEN_DIR}" \
    --save-dir "${OUT_DIR}" \
    --tokenizer-model "${HF_QWEN_DIR}" \
    --bf16

echo ""
echo "[convert] ${CKPT_TAG} done: ${OUT_DIR}"
