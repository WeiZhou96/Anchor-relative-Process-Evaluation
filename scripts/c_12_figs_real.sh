#!/usr/bin/env bash
# Figures 1 to 4 from track A's real outputs into outputs/figs.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
$PY -X utf8 report/make_figs.py
echo ---- files
ls -la outputs/figs
