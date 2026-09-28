#!/usr/bin/env bash
# Answer matrices for all nine training artefacts: j = -11..88 at delta_s = 0.25.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export CUDA_VISIBLE_DEVICES=5
MAN=data/manifest/manifest_real.csv

mkdir -p outputs/answers
for SEED in 20260903 20260904 20260905; do
  for NAME in clip__r18mean prefix__gru128 prefix__gru512; do
    $PY -X utf8 systems/infer_answers.py --manifest $MAN \
        --ckpt outputs/checkpoints/${NAME}__seed${SEED}.pt
  done
done
echo "b_05_infer done"
