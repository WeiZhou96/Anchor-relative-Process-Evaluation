#!/usr/bin/env bash
# Roll every answer matrix into one selection-proxy table, on dev and on test.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
MAN=data/manifest/manifest_real.csv

$PY -X utf8 systems/summarize.py --manifest $MAN --split dev  --out outputs/b_summary_dev.csv
$PY -X utf8 systems/summarize.py --manifest $MAN --split test --out outputs/b_summary_test.csv
echo "b_10_summary done"
