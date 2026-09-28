#!/usr/bin/env bash
# Detach a_full.sh so the ssh call returns immediately.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
mkdir -p ${APE_TMP}
nohup bash scripts/a_full.sh > ${APE_TMP}/a_full.nohup 2>&1 &
PID=$!
echo "launched a_full.sh pid $PID"
echo "log: ${APE_TMP}/a_full.log"
