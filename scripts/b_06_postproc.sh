#!/usr/bin/env bash
# Post-processing arms on each seed's GRU-512 base.
# Pass a seed as $1 to run one seed only; no argument runs all three.
# Stride 1 sweeps the grid on dev at the frozen H; strides 2 and 4 regenerate the
# SAME selected rule on a coarser grid (they are not sub-samples).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
MAN=data/manifest/manifest_real.csv
SEEDS=${1:-"20260903 20260904 20260905"}

for SEED in $SEEDS; do
  BASE=prefix__gru512__seed${SEED}
  for STRIDE in 1 2 4; do
    echo "=== postproc base=$BASE stride=$STRIDE ==="
    $PY -X utf8 systems/postproc.py --manifest $MAN --base $BASE --stride $STRIDE
  done
done
echo "b_06_postproc done"
