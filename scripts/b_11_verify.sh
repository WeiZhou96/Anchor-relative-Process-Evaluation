#!/usr/bin/env bash
# Acceptance check on the answer matrices this track owns.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
MAN=data/manifest/manifest_real.csv

$PY -X utf8 systems/verify_answers.py --manifest $MAN --only-mine
echo "b_11_verify done"
