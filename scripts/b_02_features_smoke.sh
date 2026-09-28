#!/usr/bin/env bash
# Throughput smoke: 50 dev clips, one GPU, foreground.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export CUDA_VISIBLE_DEVICES=5

mkdir -p outputs/features
$PY -X utf8 systems/features.py --splits dev --limit 50 --workers 8
echo "b_02_features_smoke done"
