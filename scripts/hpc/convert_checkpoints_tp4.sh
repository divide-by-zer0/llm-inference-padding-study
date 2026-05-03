#!/usr/bin/env bash
# Convert only the Config A checkpoint: TP=4, PP=1.
#
# Quota-friendly workflow:
#   bash scripts/hpc/convert_checkpoints_tp4.sh
#   bash scripts/hpc/submit_all_tp4.sh
#   rm -rf "$HOME/scratch.cmsc828/qwen-checkpoints/qwen2.5-32b-mcore-tp4"

set -euo pipefail

export CKPT_TAG="tp4"
export TARGET_TP=4
export TARGET_PP=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/convert_checkpoint_one.sh"
