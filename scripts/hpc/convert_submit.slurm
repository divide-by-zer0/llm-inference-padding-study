#!/usr/bin/env bash
# convert_submit.slurm — Slurm wrapper for one convert_checkpoints_*.sh script.
#
# Submits one HF→Megatron checkpoint conversion as a batch job, so you don't
# have to babysit an `salloc`.
#
# Usage:
#   sbatch --mem=240G \
#       --export=ALL,CONVERT_SCRIPT=scripts/hpc/convert_checkpoints_tp4.sh \
#       --job-name=qwen-convert-tp4 scripts/hpc/convert_submit.slurm
#
# Tail progress:
#   tail -f logs/qwen-convert-<jobid>.out
#
#SBATCH --job-name=qwen-convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cadwani@umd.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# Modules + venv  (same pattern as zaratan_submit.slurm)
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
unset PYTHONPATH

# Transformer Engine imports during checkpoint conversion.  On Zaratan, NVRTC
# may be available through modules or pip wheel paths without being listed by
# `ldconfig -p`, which TE 1.13 queries directly.  Add the venv's bundled NVIDIA
# libs to LD_LIBRARY_PATH, then provide a job-local ldconfig shim that appends
# the NVRTC entry TE expects.
NVIDIA_LIB_DIRS="$(find "${VENV_DIR}/lib/python3.10/site-packages/nvidia" \
    -name "lib" -type d 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:-}"

NVRTC_LIB="$(find "${VENV_DIR}/lib/python3.10/site-packages/nvidia" \
    -name "libnvrtc.so*" -type f 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -z "${NVRTC_LIB}" && -n "${CUDA_HOME:-}" ]]; then
    NVRTC_LIB="$(find "${CUDA_HOME}" -name "libnvrtc.so*" -type f 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [[ -n "${NVRTC_LIB}" ]]; then
    NVRTC_DIR="$(dirname "${NVRTC_LIB}")"
    export LD_LIBRARY_PATH="${NVRTC_DIR}:${LD_LIBRARY_PATH:-}"

    LDCONFIG_SHIM_DIR="${TMPDIR:-/tmp}/te-ldconfig-${SLURM_JOB_ID:-$$}"
    mkdir -p "${LDCONFIG_SHIM_DIR}"
    REAL_LDCONFIG="$(command -v ldconfig || true)"
    cat > "${LDCONFIG_SHIM_DIR}/ldconfig" <<EOF
#!/usr/bin/env bash
if [[ "\${1:-}" == "-p" ]]; then
    [[ -n "${REAL_LDCONFIG}" ]] && "${REAL_LDCONFIG}" -p 2>/dev/null || true
    echo "libnvrtc.so.12 (libc6,x86-64) => ${NVRTC_LIB}"
else
    [[ -n "${REAL_LDCONFIG}" ]] && exec "${REAL_LDCONFIG}" "\$@"
    exit 0
fi
EOF
    chmod +x "${LDCONFIG_SHIM_DIR}/ldconfig"
    export PATH="${LDCONFIG_SHIM_DIR}:${PATH}"
fi

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
export PROJECT_BASE="${PROJECT_BASE:-$HOME/scratch.cmsc828}"
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/llm-inference-padding-study}"
export MEGATRON_DIR="${MEGATRON_DIR:-$PROJECT_BASE/Megatron-LM}"
export HF_QWEN_DIR="${HF_QWEN_DIR:-$PROJECT_BASE/qwen2.5-32b-hf}"
export CKPT_ROOT="${CKPT_ROOT:-$PROJECT_BASE/qwen-checkpoints}"
export CONVERT_SCRIPT="${CONVERT_SCRIPT:-}"

# ---------------------------------------------------------------------------
# Sanity checks before the GPU clock starts ticking
# ---------------------------------------------------------------------------
[[ -d "${PROJECT_ROOT}" ]]   || { echo "PROJECT_ROOT missing: ${PROJECT_ROOT}"; exit 1; }
[[ -d "${MEGATRON_DIR}" ]]   || { echo "MEGATRON_DIR missing: ${MEGATRON_DIR}"; exit 1; }
[[ -d "${HF_QWEN_DIR}" ]]    || { echo "HF_QWEN_DIR missing: ${HF_QWEN_DIR}"; exit 1; }
[[ -n "${CONVERT_SCRIPT}" ]] || {
    echo "CONVERT_SCRIPT must be set to one of:"
    echo "  scripts/hpc/convert_checkpoints_tp4.sh"
    echo "  scripts/hpc/convert_checkpoints_tp2pp2.sh"
    echo "  scripts/hpc/convert_checkpoints_tp1pp4.sh"
    exit 1
}

if [[ "${CONVERT_SCRIPT}" = /* ]]; then
    CONVERT_SCRIPT_PATH="${CONVERT_SCRIPT}"
else
    CONVERT_SCRIPT_PATH="${PROJECT_ROOT}/${CONVERT_SCRIPT}"
fi
[[ -f "${CONVERT_SCRIPT_PATH}" ]] || { echo "CONVERT_SCRIPT missing: ${CONVERT_SCRIPT_PATH}"; exit 1; }

mkdir -p "${CKPT_ROOT}"

echo "==========================================================="
echo "  Job:        ${SLURM_JOB_NAME:-qwen-convert} (${SLURM_JOB_ID:-?})"
echo "  Started:    $(date)"
echo "  Megatron:   ${MEGATRON_DIR}"
echo "  HF model:   ${HF_QWEN_DIR}"
echo "  Output:     ${CKPT_ROOT}"
echo "  Script:     ${CONVERT_SCRIPT}"
echo "  NVRTC:      ${NVRTC_LIB:-not found}"
echo "==========================================================="

cd "${PROJECT_ROOT}"
bash "${CONVERT_SCRIPT_PATH}"

echo ""
echo "==========================================================="
echo "  Finished:   $(date)"
echo "==========================================================="
