#!/usr/bin/env bash
# Manifest statistics, attrition table, H-tier candidates and the data_*.png figures.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

$PY -X utf8 data/stats.py
echo ----
cat data/manifest/stats.md
