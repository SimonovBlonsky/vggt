#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python "$SCRIPT_DIR/eval_relpose.py" \
  --dataset scannet1500 \
  --batch-size 8 \
  "$@"
