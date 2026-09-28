#!/usr/bin/env bash
# Run the ape/ unit tests. CPU only, no video, no GPU, a few seconds.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

mkdir -p ${APE_TMP}
LOG=${APE_TMP}/a_test.log
{
  echo "=== a_test $(date -Is) ==="
  $PY -m pytest tests -q -rs --durations=10
  echo "=== a_test done $(date -Is) ==="
} 2>&1 | tee "$LOG"
