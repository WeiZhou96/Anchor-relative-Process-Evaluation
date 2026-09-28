#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

echo "=== torch hub checkpoint cache ==="
ls -la ${TORCH_HOME}/hub/checkpoints/ || echo "no torch hub cache dir"
echo "=== python probe ==="
$PY -X utf8 systems/_probe.py
echo "=== done ==="
