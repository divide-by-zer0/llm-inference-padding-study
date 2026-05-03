#!/usr/bin/env bash
# Submit only Config C, which uses the qwen2.5-32b-mcore-tp1pp4 checkpoint.
#
# Run from the project root:
#   bash scripts/hpc/submit_all_tp1pp4.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs results

CFG="c"
JOB_NAME="pad-${CFG}-tp1pp4"

echo "Submitting Config ${CFG} from ${PROJECT_ROOT}"
echo "Checkpoint tag: tp1pp4"
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
