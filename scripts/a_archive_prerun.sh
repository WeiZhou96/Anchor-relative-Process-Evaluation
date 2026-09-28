#!/usr/bin/env bash
# Move the previous run's products aside before re-running.
#
# The destination is AUTO-VERSIONED (_prerun_v1, _prerun_v2, ...). An earlier
# version of this script wrote to a fixed _prerun_v1 and clobbered the previous
# archive when it was run a second time; archives are audit trail and must never
# be overwritten, so the next free slot is chosen instead.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
LOG=${APE_TMP}/a_archive_prerun.log

N=1
while [ -e "outputs/_prerun_v${N}" ]; do
  N=$((N + 1))
done
DEST="outputs/_prerun_v${N}"

{
  echo "=== a_archive_prerun $(date -Is) ==="
  echo "destination (next free slot): $DEST"
  mkdir -p "$DEST"
  MOVED=0
  for item in metrics calib calib_plaus phenomena prefix_lists _prefreeze_stale \
              report_index.json report_index.csv; do
    if [ -e "outputs/$item" ]; then
      mv "outputs/$item" "$DEST/$item"
      echo "archived outputs/$item -> $DEST/$item"
      MOVED=$((MOVED + 1))
    else
      echo "absent (nothing to archive): outputs/$item"
    fi
  done
  if [ -f ${APE_TMP}/a_full.log ]; then
    cp ${APE_TMP}/a_full.log "$DEST/a_full.log"
    echo "archived a_full.log -> $DEST/a_full.log"
  fi
  if [ "$MOVED" -eq 0 ]; then
    echo "nothing was archived; removing the empty slot"
    rmdir "$DEST" 2>/dev/null || true
  fi
  echo "--- outputs/ after archiving ---"
  ls -1 outputs/
  echo "=== a_archive_prerun done $(date -Is) ==="
} 2>&1 | tee "$LOG"
