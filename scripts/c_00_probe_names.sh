#!/usr/bin/env bash
# One-off probe of clip file-name structure, used to pick the source-cluster rule.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

$PY -X utf8 data/_probe_names.py
