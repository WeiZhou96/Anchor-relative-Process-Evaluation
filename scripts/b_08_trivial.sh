#!/usr/bin/env bash
# Trivial systems. Restricted to clips with a cached feature file so that every
# system in the library covers exactly the same clip set.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
MAN=data/manifest/manifest_real.csv

$PY -X utf8 systems/trivial.py --manifest $MAN --restrict-to-features
echo "b_08_trivial done"
