#!/usr/bin/env bash
# Temporary B-track manifest straight from metadata-real.csv (CONTRACT 5.1 columns).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

mkdir -p outputs
$PY -X utf8 systems/tmp_manifest.py --out outputs/tmp_manifest_b.csv
echo "b_01_manifest done"
