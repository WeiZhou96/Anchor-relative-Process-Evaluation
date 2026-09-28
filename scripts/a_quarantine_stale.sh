#!/usr/bin/env bash
# Move pre-freeze artefacts out of outputs/ so the report index cannot mix
# protocol versions. Archived, never deleted: they are part of the audit trail.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export PYTHONUNBUFFERED=1
LOG=${APE_TMP}/a_quarantine.log
STALE=outputs/_prefreeze_stale
{
  echo "=== a_quarantine_stale $(date -Is) ==="
  mkdir -p "$STALE"
  $PY -c "
import json, os, shutil, glob
from ape.protocol import load_protocol_checked
cfg = load_protocol_checked('protocol/pi0.yaml')
current = {cfg.pi0(h).pi_hash for h in cfg.h_list_s}
print('current frozen pi_hashes:', sorted(current))
moved = []
for root in ('outputs/metrics', 'outputs/calib', 'outputs/calib_plaus', 'outputs/phenomena'):
    if not os.path.isdir(root):
        continue
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p) or d in current:
            continue
        dest = os.path.join('$STALE', root.split('/')[-1] + '__' + d)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(p, dest)
        moved.append((p, dest))
for a, b in moved:
    print('quarantined:', a, '->', b)
print('total quarantined dirs:', len(moved))
"
  echo "--- remaining dirs (must be only the frozen hashes) ---"
  for r in metrics calib calib_plaus phenomena; do
    echo "outputs/$r: $(ls outputs/$r 2>/dev/null | tr '\n' ' ')"
  done
  echo "--- rebuild the report index from the frozen results only ---"
  $PY -m ape.cli report-json --metrics outputs/metrics --out outputs/report_index.json
  echo "=== a_quarantine_stale done $(date -Is) ==="
} 2>&1 | tee "$LOG"
