#!/usr/bin/env bash
# Manifest v2: keep the v1 file under its own name, then rebuild with the leak repair.
# v1 = official split verbatim. v2 = no source cluster straddles a split; the audit set
# stays a strict subset of the official test split; the dev subset is NOT re-drawn.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

M=data/manifest
if [ -f $M/manifest_real.csv ] && [ ! -f $M/manifest_real_v1_official.csv ]; then
  cp $M/manifest_real.csv $M/manifest_real_v1_official.csv
  echo "preserved v1 as $M/manifest_real_v1_official.csv"
fi

$PY -X utf8 data/build_manifest.py --kind real
echo ---- v1 md5
md5sum $M/manifest_real_v1_official.csv
echo ---- v2 md5
md5sum $M/manifest_real.csv
echo ---- header
head -n 1 $M/manifest_real.csv
