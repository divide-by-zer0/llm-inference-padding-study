#!/usr/bin/env bash
# pull_assets.sh — fetch the two large external assets we depend on:
#
#   1. Megatron-LM repo (the inference framework + checkpoint conversion tools)
#   2. Qwen2.5-32B HF safetensors (~65 GB, the model weights)
#
# Both go to user-configurable locations (defaults below).  Run this once on
# a Zaratan login node before convert_checkpoints.sh.
#
# Usage:
#   bash scripts/hpc/pull_assets.sh
#
# Override defaults via env vars:
#   MEGATRON_DIR=$HOME/code/Megatron-LM \
#   HF_QWEN_DIR=$SCRATCH/models/qwen2.5-32b-hf \
#   bash scripts/hpc/pull_assets.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# USER CONFIG — verify / override
# ---------------------------------------------------------------------------
MEGATRON_DIR="${MEGATRON_DIR:-$HOME/Megatron-LM}"
HF_QWEN_DIR="${HF_QWEN_DIR:-$SCRATCH/qwen2.5-32b-hf}"

# Megatron-LM commit to pin to (for reproducibility).  Leave blank to use main.
# Pick a recent stable commit hash if you want a frozen reference.
MEGATRON_COMMIT="${MEGATRON_COMMIT:-}"

# HF model ID
HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen2.5-32B}"

# ---------------------------------------------------------------------------
# 1. Megatron-LM
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[pull_assets] Megatron-LM  →  ${MEGATRON_DIR}"
echo "============================================================"

if [[ -d "${MEGATRON_DIR}/.git" ]]; then
    echo "[megatron] Already cloned; pulling latest …"
    git -C "${MEGATRON_DIR}" fetch --tags --quiet
    git -C "${MEGATRON_DIR}" pull --ff-only || {
        echo "[megatron] WARNING: pull failed (local edits?). Skipping update."
    }
else
    git clone https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_DIR}"
fi

if [[ -n "${MEGATRON_COMMIT}" ]]; then
    echo "[megatron] Checking out pinned commit ${MEGATRON_COMMIT}"
    git -C "${MEGATRON_DIR}" checkout "${MEGATRON_COMMIT}"
fi

# Sanity check: the checkpoint conversion tool we need
if [[ ! -f "${MEGATRON_DIR}/tools/checkpoint/loader_llama_mistral.py" ]]; then
    echo "ERROR: loader_llama_mistral.py not found in this Megatron-LM checkout."
    echo "       The conversion script depends on it.  Try a different commit/branch."
    exit 1
fi
echo "[megatron] OK."

# ---------------------------------------------------------------------------
# 2. Qwen2.5-32B safetensors
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[pull_assets] ${HF_MODEL_ID}  →  ${HF_QWEN_DIR}"
echo "============================================================"

mkdir -p "${HF_QWEN_DIR}"

# Sanity check: are the safetensors already there?
if compgen -G "${HF_QWEN_DIR}/*.safetensors" > /dev/null; then
    echo "[hf] ${HF_QWEN_DIR} already contains safetensors; skipping download."
else
    # huggingface-cli must be on PATH (installed by setup_env.sh).
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "ERROR: huggingface-cli not found.  Run setup_env.sh first to install it."
        exit 1
    fi

    # Use hf_transfer for fast parallel download (~3-5x faster on HPC networks).
    # Falls back to default downloader if the package isn't installed.
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

    echo "[hf] Downloading (this is ~65 GB; may take 10-30 min depending on link)…"
    huggingface-cli download "${HF_MODEL_ID}" \
        --local-dir "${HF_QWEN_DIR}" \
        --local-dir-use-symlinks False \
        --resume-download
fi

# Sanity check: tokenizer + at least one safetensor shard
[[ -f "${HF_QWEN_DIR}/tokenizer.json" ]] || {
    echo "ERROR: tokenizer.json missing in ${HF_QWEN_DIR}"; exit 1;
}
[[ -f "${HF_QWEN_DIR}/config.json" ]] || {
    echo "ERROR: config.json missing in ${HF_QWEN_DIR}"; exit 1;
}
SHARD_COUNT=$(ls "${HF_QWEN_DIR}"/model-*.safetensors 2>/dev/null | wc -l)
if [[ "${SHARD_COUNT}" -lt 1 ]]; then
    echo "ERROR: no model-*.safetensors shards found in ${HF_QWEN_DIR}"
    exit 1
fi
echo "[hf] OK.  ${SHARD_COUNT} safetensor shard(s)."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[pull_assets] Done."
echo "  MEGATRON_DIR = ${MEGATRON_DIR}"
echo "  HF_QWEN_DIR  = ${HF_QWEN_DIR}"
echo ""
echo "Next step:  bash scripts/hpc/convert_checkpoints.sh"
echo "============================================================"
