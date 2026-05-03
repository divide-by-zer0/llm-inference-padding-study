#!/usr/bin/env bash
# Submit only Config B, which uses the qwen2.5-32b-mcore-tp2pp2 checkpoint.
#
# Run from the project root:
#   bash scripts/hpc/submit_all_tp2pp2.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs results

CFG="b"
JOB_NAME="pad-${CFG}-tp2pp2"

echo "Submitting Config ${CFG} from ${PROJECT_ROOT}"
echo "Checkpoint tag: tp2pp2"
echo "-> sbatch CONFIG=${CFG} (job-name=${JOB_NAME})"

sbatch \
    --job-name="${JOB_NAME}" \
    --export=ALL,CONFIG="${CFG}" \
    scripts/hpc/zaratan_submit.slurm

echo ""
echo "Queue status:"
squeue -u "$USER"

echo ""
echo "Tail the job log with:"
echo "    tail -f /home/cadwani/padding-results/logs/${JOB_NAME}-*.out"
