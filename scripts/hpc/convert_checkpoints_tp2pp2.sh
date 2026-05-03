#!/usr/bin/env bash
# Convert only the Config B checkpoint: TP=2, PP=2.
#
# Quota-friendly workflow:
#   bash scripts/hpc/convert_checkpoints_tp2pp2.sh
#   bash scripts/hpc/submit_all_tp2pp2.sh
#   rm -rf "$HOME/scratch.cmsc828/qwen-checkpoints/qwen2.5-32b-mcore-tp2pp2"

set -euo pipefail

export CKPT_TAG="tp2pp2"
export TARGET_TP=2
export TARGET_PP=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/convert_checkpoint_one.sh"
