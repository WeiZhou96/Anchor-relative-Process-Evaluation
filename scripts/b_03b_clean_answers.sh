#!/usr/bin/env bash
# Remove only THIS track's answer directories before the S1 rebuild, so that a
# parameter that is no longer selected cannot leave a stale system behind.
# Touches nothing else: the A track's block__* directories are left alone, and the
# deletion is restricted to outputs/answers by an explicit cd.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}/outputs/answers

for PREFIX in clip__ prefix__ postproc__ commit__ trivial__; do
  for D in ${PREFIX}*; do
    if [ -d "$D" ]; then
      echo "removing $D"
      rm -rf -- "$D"
    fi
  done
done
for F in _commit_dev_scan__*.csv; do
  if [ -f "$F" ]; then
    echo "removing $F"
    rm -f -- "$F"
  fi
done

echo "--- remaining entries ---"
ls
echo "b_03b_clean_answers done"
