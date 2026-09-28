#!/usr/bin/env bash
# Full feature extraction, backgrounded. Log: ${APE_TMP}/b_features.log
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export CUDA_VISIBLE_DEVICES=5

mkdir -p outputs/features ${APE_TMP}
LOG=${APE_TMP}/b_features.log
nohup $PY -X utf8 systems/features.py --workers 12 > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > ${APE_TMP}/b_features.pid
disown || true
sleep 2
echo "launched pid $PID; log $LOG"
