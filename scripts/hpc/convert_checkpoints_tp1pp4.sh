#!/usr/bin/env bash
# Convert only the Config C checkpoint: TP=1, PP=4.
#
# Quota-friendly workflow:
#   bash scripts/hpc/convert_checkpoints_tp1pp4.sh
#   bash scripts/hpc/submit_all_tp1pp4.sh
#   rm -rf "$HOME/scratch.cmsc828/qwen-checkpoints/qwen2.5-32b-mcore-tp1pp4"

set -euo pipefail

export CKPT_TAG="tp1pp4"
export TARGET_TP=1
export TARGET_PP=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/convert_checkpoint_one.sh"
