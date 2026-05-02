#!/usr/bin/env bash
# setup_env.sh — one-time Python environment setup on Zaratan
#
# Run this ONCE on a Zaratan login node (no GPUs needed).  It loads the
# required modules, creates a conda env, installs all Python deps, and
# verifies that Megatron-LM imports correctly.
#
# Usage:
#   bash scripts/hpc/setup_env.sh
#
# Before running, fill in the placeholder paths in the "USER CONFIG" block.

set -euo pipefail

# ---------------------------------------------------------------------------
# USER CONFIG — verify / fill in for Zaratan
# ---------------------------------------------------------------------------
# Path to the Megatron-LM clone on Zaratan
MEGATRON_DIR="${MEGATRON_DIR:-$HOME/Megatron-LM}"

# Conda env name
ENV_NAME="${ENV_NAME:-padding-bench}"

# CUDA version to match the available module
CUDA_VERSION="${CUDA_VERSION:-12.4}"

# ---------------------------------------------------------------------------
# Module loads
# ---------------------------------------------------------------------------
# Run `module avail cuda gcc anaconda` on Zaratan to confirm exact module names.
echo "[setup_env] Loading modules …"
module purge
module load cuda/${CUDA_VERSION}
module load gcc
module load anaconda                  # or 'miniconda3' depending on Zaratan
# module load openmpi                 # uncomment if NCCL needs it

# ---------------------------------------------------------------------------
# Conda env
# ---------------------------------------------------------------------------
echo "[setup_env] Creating conda env '${ENV_NAME}' …"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

# shellcheck disable=SC1091
source activate "${ENV_NAME}"

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
echo "[setup_env] Installing PyTorch + dependencies …"

# PyTorch — pick the wheel that matches the CUDA version on Zaratan.
# CUDA 12.4 → cu124; CUDA 12.1 → cu121; etc.
CUDA_TAG="cu$(echo "${CUDA_VERSION}" | tr -d .)"
pip install --upgrade pip
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# Megatron + benchmark dependencies
pip install \
    transformers \
    datasets \
    accelerate \
    sentencepiece \
    pandas \
    matplotlib \
    numpy \
    pyyaml \
    pybind11 \
    six \
    "huggingface_hub[cli]" \
    hf_transfer

# Megatron-LM may also need flash-attn / transformer-engine.
# These are heavy and CUDA-specific; install only if your launch scripts use
# `--transformer-impl transformer_engine` (they do).
pip install transformer-engine[pytorch] || \
    echo "[setup_env] WARNING: transformer-engine install failed; you may need to build from source"

# ---------------------------------------------------------------------------
# Verify Megatron-LM is importable
# ---------------------------------------------------------------------------
echo "[setup_env] Verifying imports …"
python - <<PY
import sys, os, torch
print(f"  torch              {torch.__version__}")
print(f"  CUDA available?    {torch.cuda.is_available()}")
sys.path.insert(0, os.environ.get("MEGATRON_DIR", "${MEGATRON_DIR}"))
import megatron
print(f"  megatron import    OK  ({megatron.__file__})")
PY

echo ""
echo "[setup_env] Done."
echo "Activate the env in future sessions with:"
echo "    module load cuda/${CUDA_VERSION} gcc anaconda"
echo "    source activate ${ENV_NAME}"
