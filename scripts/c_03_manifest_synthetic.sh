#!/usr/bin/env bash
# Registration-only manifest for the synthetic (CARLA) subset: no md5, no decode probe.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

$PY -X utf8 data/build_manifest.py --kind synthetic
head -n 3 data/manifest/manifest_synthetic.csv
wc -l data/manifest/manifest_synthetic.csv
