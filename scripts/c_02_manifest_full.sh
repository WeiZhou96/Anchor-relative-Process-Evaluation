#!/usr/bin/env bash
# Full real manifest (2027 clips) with md5 and OpenCV decode probe, launched in the
# background because hashing 4.6 GB of video takes minutes. Log:
#   ${APE_TMP}/c_manifest.log
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

LOG=${APE_TMP}/c_manifest.log
rm -f "$LOG"
nohup $PY -X utf8 data/build_manifest.py --kind real > "$LOG" 2>&1 &
echo "launched pid $!"
sleep 5
tail -n 5 "$LOG"
