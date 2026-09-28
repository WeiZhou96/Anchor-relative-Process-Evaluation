#!/usr/bin/env bash
# Coarse-step rebuilds of the random trivial system, the same way the post-processing
# and commitment arms get theirs: the draws are per grid point, so a coarser delta_s
# is a regenerated system, not a sub-sample. --only random keeps the other three
# trivial directories untouched.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
MAN=data/manifest/manifest_real.csv

for STRIDE in 2 4; do
  echo "=== trivial random stride=$STRIDE ==="
  $PY -X utf8 systems/trivial.py --manifest $MAN --restrict-to-features \
      --only random --stride $STRIDE
done
echo "b_08b_trivial_stride done"
