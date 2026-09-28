#!/usr/bin/env bash
# B-side checks: array-logic self test, then sub-sampling equivalence on real
# checkpoints and the real feature cache (one seed is enough for the equivalence
# argument, which is about the caching scheme rather than the weights).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export CUDA_VISIBLE_DEVICES=5
MAN=data/manifest/manifest_real.csv

$PY -X utf8 systems/_selftest.py
for S in 2 4; do
  $PY -X utf8 systems/check_subsample.py --manifest $MAN \
      --ckpt outputs/checkpoints/clip__r18mean__seed20260903.pt --stride $S
  $PY -X utf8 systems/check_subsample.py --manifest $MAN \
      --ckpt outputs/checkpoints/prefix__gru512__seed20260903.pt --stride $S
done
echo "b_09_checks done"
