#!/usr/bin/env bash
# submit_all.sh — submit one Slurm job per parallelism config (a, b, c).
#
# Run from the project root:
#   bash scripts/hpc/submit_all.sh
#
# Each job is independent — if one fails, the others still run.  Resubmit
# any failed config with:
#   CONFIG=a sbatch --export=ALL,CONFIG=a scripts/hpc/zaratan_submit.slurm

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs results

echo "Submitting jobs from ${PROJECT_ROOT}"
echo ""

for cfg in a b c; do
    JOB_NAME="pad-${cfg}"
    echo "→ sbatch CONFIG=${cfg} (job-name=${JOB_NAME})"
    sbatch \
        --job-name="${JOB_NAME}" \
        --export=ALL,CONFIG="${cfg}" \
        scripts/hpc/zaratan_submit.slurm
done

echo ""
echo "All jobs submitted.  Queue status:"
squeue -u "$USER"

echo ""
echo "Tail a job log with:"
echo "    tail -f logs/pad-a-*.out"
