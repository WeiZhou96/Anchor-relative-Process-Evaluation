#!/usr/bin/env bash
# Smoke run of the real manifest builder on the first 50 clips (md5 + OpenCV decode probe).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

$PY -X utf8 data/build_manifest.py --kind real --limit 50 --out ${APE_TMP}/manifest_real_smoke.csv
head -n 3 ${APE_TMP}/manifest_real_smoke.csv
wc -l ${APE_TMP}/manifest_real_smoke.csv
