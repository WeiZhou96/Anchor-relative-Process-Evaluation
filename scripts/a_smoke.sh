#!/usr/bin/env bash
# Full-chain smoke of the protocol library on track C's real manifest:
#   make-prefixes -> blocks -> stand-ins -> eval -> scan -> phenomena -> report-json
#
# Reads only the manifest CSV and the answer matrices it generates itself: no
# video is decoded, no model is run, no GPU is used. The bootstrap budget is cut
# to 200 so the smoke stays quick; the frozen value in protocol/pi0.yaml is 1000
# and the reportable run must use it.
#
# The stand-in answer matrices under outputs/answers_a_standin/ are placeholders
# that exist only to give the pair-testing and rank-preservation code something
# with per-video variation before track B's systems land. They are NOT systems
# and their numbers are not reportable.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
# stdout goes through tee, so Python would block-buffer its progress lines and the
# log would look stalled for minutes at a time; keep the run observable
export PYTHONUNBUFFERED=1

mkdir -p ${APE_TMP}
LOG=${APE_TMP}/a_smoke.log

MANIFEST=data/manifest/manifest_real.csv
if [ ! -f "$MANIFEST" ]; then
  MANIFEST=tests/fixtures/manifest_min.csv
fi
PI=protocol/pi0.yaml
BLOCKS=outputs/answers
STANDIN=outputs/answers_a_standin
BOOT=200

{
  echo "=== a_smoke $(date -Is) ==="
  echo "manifest: $MANIFEST"
  echo "protocol: $PI"
  $PY -c "
import sys
from ape.protocol import load_manifest, load_protocol
from ape.cohort import check_cluster_rule, cohort_table
cfg = load_protocol('$PI')
m = load_manifest('$MANIFEST')
print('rows after decode_ok filter:', len(m))
bad = check_cluster_rule(m)
print('cluster-rule mismatches:', len(bad))
if len(bad):
    print(bad.head(10).to_string())
print(cohort_table(m, cfg.h_list_s).to_string())
print('delta_fine_s', cfg.delta_fine_s, 'pad_s', cfg.answers_pad_s, 'grid_max_s', cfg.grid_max_s)
"

  echo "--- 1/7 make-prefixes ---"
  time $PY -m ape.cli make-prefixes --pi "$PI" --manifest "$MANIFEST" \
      --out outputs/prefix_lists

  echo "--- 2/7 blocks ---"
  time $PY -m ape.cli blocks --pi "$PI" --manifest "$MANIFEST" --out "$BLOCKS"

  echo "--- 3/7 stand-in answer matrices (placeholders, not systems) ---"
  # Stand-ins exist only to give the pair/rank code something to chew on before
  # track B's systems land. Once any non-block system is present in
  # outputs/answers they are dropped: leaving them in would put placeholder
  # matrices into P and contaminate R_M.
  N_REAL=$($PY -c "
import glob, os
n = 0
for d in sorted(glob.glob('outputs/answers/*')):
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'answers.csv')):
        if not os.path.basename(d).startswith('block__'):
            n += 1
print(n)
")
  echo "non-block systems already in outputs/answers: $N_REAL"
  if [ "$N_REAL" -gt 0 ]; then
    echo "track B systems present -> stand-ins EXCLUDED from this run"
    ANSWERS="$BLOCKS"
  else
    echo "no real systems yet -> generating stand-ins"
    mkdir -p "$STANDIN"
    time $PY -X utf8 tests/fixtures/make_standins.py --manifest "$MANIFEST" \
        --out "$STANDIN" --span 22.0 --pad 3.0
    ANSWERS="$BLOCKS,$STANDIN"
  fi
  echo "answers spec: $ANSWERS"

  echo "--- 4/7 eval ---"
  time $PY -m ape.cli eval --pi "$PI" --manifest "$MANIFEST" \
      --answers "$ANSWERS" --out outputs/metrics --bootstrap-n "$BOOT"

  echo "--- 5/7 scan ---"
  time $PY -m ape.cli scan --pi "$PI" --manifest "$MANIFEST" \
      --answers "$ANSWERS" --out outputs/calib --bootstrap-n "$BOOT" --mode axis

  echo "--- 6/7 phenomena ---"
  time $PY -m ape.cli phenomena --pi "$PI" --manifest "$MANIFEST" \
      --answers "$ANSWERS" --out outputs/phenomena

  echo "--- 7/7 report-json ---"
  time $PY -m ape.cli report-json --metrics outputs/metrics \
      --out outputs/report_index.json

  echo "--- artefacts ---"
  find outputs -maxdepth 2 -type d | sort
  echo "--- metric json count ---"
  find outputs/metrics -name '*.json' | wc -l
  echo "--- head of the report index ---"
  head -n 20 outputs/report_index.csv
  echo "=== a_smoke done $(date -Is) ==="
} 2>&1 | tee "$LOG"
