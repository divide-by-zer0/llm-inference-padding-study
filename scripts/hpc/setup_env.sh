#!/usr/bin/env bash
# setup_env.sh — one-time Python environment setup on Zaratan.
#
# Run this ONCE on a Zaratan login node (no GPU needed).  It loads the
# bundled Python+CUDA module, creates a venv, installs all Python deps,
# and verifies that Megatron-LM imports correctly.
#
# Usage:
#   bash scripts/hpc/setup_env.sh
#
# We use venv (not conda) because Zaratan has no anaconda module.  The
# Python module we load already includes CUDA 12.3 + gcc 11.3.0 built for
# zen2 (AMD EPYC, the architecture of Zaratan's A100 nodes).

set -euo pipefail

# ---------------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------------
MEGATRON_DIR="${MEGATRON_DIR:-$HOME/Megatron-LM}"
VENV_DIR="${VENV_DIR:-$HOME/padding-bench-venv}"

# Zaratan module that ships Python 3.10.10 + CUDA 12.3 + gcc 11.3 (zen2 build).
# Verified via `module avail python` on 2026-05-01.
PY_MODULE="${PY_MODULE:-python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2}"

# PyTorch wheel CUDA tag.  Zaratan has CUDA 12.3; PyTorch ships cu121 / cu124.
# cu121 is forward-compatible with the CUDA 12.3 driver and is the safest match.
PYTORCH_CUDA_TAG="${PYTORCH_CUDA_TAG:-cu121}"

# ---------------------------------------------------------------------------
# Module loads
# ---------------------------------------------------------------------------
echo "[setup_env] Loading modules …"
module purge
module load "${PY_MODULE}"

echo "[setup_env]   python: $(which python)  ($(python --version))"
echo "[setup_env]   gcc   : $(which gcc)     ($(gcc --version | head -1))"
echo "[setup_env]   nvcc  : $(which nvcc 2>/dev/null || echo 'not in PATH')"

# ---------------------------------------------------------------------------
# Venv
# ---------------------------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[setup_env] Creating venv at ${VENV_DIR} …"
    python -m venv "${VENV_DIR}"
else
    echo "[setup_env] Reusing existing venv at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
echo "[setup_env] Installing PyTorch (${PYTORCH_CUDA_TAG}) …"
pip install torch torchvision \
    --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_TAG}"

echo "[setup_env] Installing benchmark dependencies …"
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

# Megatron-LM uses transformer_engine (per the launch scripts' --transformer-impl flag).
# It's CUDA-specific and large.  The pip install can fail if no GPU is visible;
# in that case install it later from a compute node, or build from source.
echo "[setup_env] Installing transformer-engine (may fail on login nodes) …"
pip install transformer-engine[pytorch] || \
    echo "[setup_env] WARNING: transformer-engine install failed; retry from a GPU compute node."

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo "[setup_env] Verifying imports …"
python - <<PY
import sys, os
import torch
print(f"  torch              {torch.__version__}")
print(f"  CUDA available?    {torch.cuda.is_available()} (no GPU on login node is OK)")

megatron_dir = os.environ.get("MEGATRON_DIR", "${MEGATRON_DIR}")
if os.path.isdir(megatron_dir):
    sys.path.insert(0, megatron_dir)
    try:
        import megatron
        print(f"  megatron import    OK  ({megatron.__file__})")
    except ImportError as e:
        print(f"  megatron import    FAILED: {e}")
        print("  (run pull_assets.sh first to clone Megatron-LM)")
else:
    print(f"  megatron-lm dir    not found at {megatron_dir}")
    print("  (run pull_assets.sh next to clone it)")
PY

echo ""
echo "[setup_env] Done."
echo ""
echo "To activate this env in future sessions, run:"
echo "    module load ${PY_MODULE}"
echo "    source ${VENV_DIR}/bin/activate"
