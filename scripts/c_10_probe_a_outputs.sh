#!/usr/bin/env bash
# Probe track A's real output layout so the renderers can be written against it.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
$PY -X utf8 scripts/c_10_probe_a_outputs.py
