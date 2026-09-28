#!/usr/bin/env bash
# Tables 1 to 5 from track A's real outputs into outputs/tables.
# The directory is cleared first: horizon tags appear in the file names, so a re-freeze
# that changes the H tiers would otherwise leave stale tables from the old tags behind.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
rm -rf outputs/tables
mkdir -p outputs/tables
$PY -X utf8 report/make_tables.py
echo ---- files
ls -la outputs/tables
