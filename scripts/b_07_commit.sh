#!/usr/bin/env bash
# Commitment arms on each seed's GRU-512 base, all thresholds, three grid steps.
# Pass a seed as $1 to run one seed only; no argument runs all three.
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
    echo "=== commit base=$BASE stride=$STRIDE ==="
    $PY -X utf8 systems/commit.py --manifest $MAN --base $BASE --stride $STRIDE
  done
done
echo "b_07_commit done"
