#!/usr/bin/env bash
# Source-event cluster diagnostics: size distribution, cross-split leakage, pHash spot check.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

$PY -X utf8 data/clusters.py --pairs 100
echo ----
head -n 60 data/manifest/clusters_report.md
