#!/usr/bin/env bash
# Train the three exam-takers on each of the three frozen seeds, against the S1
# manifest (v2: train 411 / dev 102 / test 1514). Training reads split==train only;
# every hyper-parameter and the stopping epoch come from split==dev.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export CUDA_VISIBLE_DEVICES=5
MAN=data/manifest/manifest_real.csv

mkdir -p outputs/checkpoints
for SEED in 20260903 20260904 20260905; do
  echo "=== seed $SEED ==="
  $PY -X utf8 systems/train_clip.py   --manifest $MAN --seed $SEED \
      --system-id clip__r18mean__seed${SEED}
  $PY -X utf8 systems/train_prefix.py --manifest $MAN --seed $SEED --hidden 128 \
      --system-id prefix__gru128__seed${SEED}
  $PY -X utf8 systems/train_prefix.py --manifest $MAN --seed $SEED --hidden 512 \
      --system-id prefix__gru512__seed${SEED}
done
echo "b_04_train done"
