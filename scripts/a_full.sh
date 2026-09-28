#!/usr/bin/env bash
# Full audit run on the frozen protocol (S1, protocol_version 1.0-S1-2026-09-03).
#
#   bash scripts/a_full.sh                 # run in the foreground
#   nohup bash scripts/a_full.sh &         # run detached; log below either way
#   tail -f ${APE_TMP}/a_full.log
#
# Stages: eval (3 horizons) -> axis scan -> plausible-subgrid product scan ->
#         phenomena -> report index.
#
# Everything uses the frozen bootstrap_n (1000); no --bootstrap-n override is
# passed, so the intervals are the frozen protocol's intervals. The CLI refuses
# to start if protocol/pi0.yaml has drifted from protocol/pi0.frozen.sha256.
#
# Stand-ins: outputs/answers_a_standin/ is deleted, not merely skipped. Those
# were placeholders for track B's systems and must never enter P.
#
# Per-step re-runs: a system whose system_card.yaml sets
# subsampling_equivalent: false is not sub-sampled on the step axis; the CLI
# reads <system_id>__stride2 and <system_id>__stride4 instead (suffix comes from
# answers.stride_suffix in pi0.yaml). If a declared re-run is missing, the run
# continues but records a note in calibration.json / _pairs.json rather than
# silently sub-sampling a system that declared that invalid.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export PYTHONUNBUFFERED=1

mkdir -p ${APE_TMP}
LOG=${APE_TMP}/a_full.log

MANIFEST=${APE_MANIFEST:-data/manifest/manifest_real.csv}
PI=${APE_PI:-protocol/pi0.yaml}
ANSWERS=${APE_ANSWERS:-outputs/answers}
SPLIT=${APE_SPLIT:-test}   # the audit set; anything else is diagnostic only
OUT=${APE_OUT:-outputs}

{
  echo "=== a_full $(date -Is) ==="
  echo "manifest : $MANIFEST"
  echo "protocol : $PI"
  echo "answers  : $ANSWERS"
  echo "split    : $SPLIT (audit set = test)"
  echo "out      : $OUT"

  if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: manifest not found: $MANIFEST"
    exit 1
  fi

  # stand-ins are placeholders; they must not reach the audit
  if [ -d outputs/answers_a_standin ]; then
    echo "removing stand-in placeholder matrices: outputs/answers_a_standin"
    rm -rf outputs/answers_a_standin
  fi

  echo "--- freeze check and inventory ---"
  $PY -c "
from ape.protocol import load_protocol_checked, content_sha256, frozen_hash_path
from ape.protocol import load_manifest
from ape.cohort import check_cluster_rule, cohort_table
import glob, os, json

cfg = load_protocol_checked('$PI')
print('protocol_version:', cfg.protocol_version, '| frozen:', cfg.frozen)
print('sha256          :', content_sha256('$PI'))
print('frozen_by       :', cfg.raw.get('frozen_by'), '| frozen_at:', cfg.raw.get('frozen_at'))
print('manifest_md5    :', cfg.raw.get('manifest_md5'), '| dev_hash:', cfg.raw.get('dev_hash'))
print('bootstrap_n     :', cfg.bootstrap_n, '| r0:', cfg.r0, '| alpha:', cfg.alpha_pairs)
print('h_list_s        :', cfg.h_list_s, '| grid_max_s:', cfg.grid_max_s)
print('delta_s         :', cfg.delta_s, '| delta_fine_s:', cfg.delta_fine_s, '| pad_s:', cfg.answers_pad_s)
for h in cfg.h_list_s:
    print('  pi_hash H=%-6g %s' % (h, cfg.pi0(h).pi_hash))

m = load_manifest('$MANIFEST')
print('manifest rows (decode_ok):', len(m))
bad = check_cluster_rule(m)
print('cluster-rule mismatches  :', len(bad))
if len(bad):
    print(bad.head(10).to_string())
from ape.cohort import select_split, split_summary
print('splits:', split_summary(m))
audit = select_split(m, '$SPLIT')
print('audit split %r -> %d clips, %d clusters' % (
    '$SPLIT', len(audit), audit['source_cluster_id'].nunique()))
print(cohort_table(audit, cfg.h_list_s).to_string())

suffix = str(cfg.answers_cfg.get('stride_suffix', '__stride'))
dirs = [d for d in sorted(glob.glob('$ANSWERS/*')) if os.path.isdir(d)
        and os.path.exists(os.path.join(d, 'answers.csv'))]
base = [d for d in dirs if suffix not in os.path.basename(d)]
stride = [d for d in dirs if suffix in os.path.basename(d)]
print('answer dirs: %d total, %d base, %d per-step re-runs' % (len(dirs), len(base), len(stride)))
need = []
import yaml
for d in base:
    c = os.path.join(d, 'system_card.yaml')
    card = yaml.safe_load(open(c)) if os.path.exists(c) else {}
    if str((card or {}).get('subsampling_equivalent', '')).lower() in ('false','0','no'):
        need.append(os.path.basename(d))
print('systems declaring subsampling_equivalent=false:', need if need else 'none')
missing = [s + suffix + str(f) for s in need for f in (2, 4)
           if not os.path.isdir(os.path.join('$ANSWERS', s + suffix + str(f)))]
print('missing per-step re-runs:', missing if missing else 'none')
"

  echo "--- 1/6 prefix lists, published artefact for external reuse ---"
  time $PY -m ape.cli make-prefixes --pi "$PI" --manifest "$MANIFEST" --split "$SPLIT" \
      --out "$OUT/prefix_lists"

  echo "--- 2/6 eval, three horizons, frozen bootstrap ---"
  time $PY -m ape.cli eval --pi "$PI" --manifest "$MANIFEST" --split "$SPLIT" \
      --answers "$ANSWERS" --out "$OUT/metrics"

  echo "--- 3/6 scan, axis mode (propagation curves for E4) ---"
  time $PY -m ape.cli scan --pi "$PI" --manifest "$MANIFEST" --split "$SPLIT" \
      --answers "$ANSWERS" --out "$OUT/calib" --mode axis

  echo "--- 4/6 scan, plausible subgrid, full product (comparability region) ---"
  time $PY -m ape.cli scan --pi "$PI" --manifest "$MANIFEST" --split "$SPLIT" \
      --answers "$ANSWERS" --out "$OUT/calib_plaus" --mode full --plaus

  echo "--- 5/6 phenomena ---"
  time $PY -m ape.cli phenomena --pi "$PI" --manifest "$MANIFEST" --split "$SPLIT" \
      --answers "$ANSWERS" --out "$OUT/phenomena"

  echo "--- 6/6 report index ---"
  time $PY -m ape.cli report-json --metrics "$OUT/metrics" \
      --out "$OUT/report_index.json"

  echo "--- artefacts ---"
  find "$OUT" -maxdepth 2 -type d | sort
  echo "metric json files: $(find "$OUT/metrics" -name '*.json' | wc -l)"
  echo "--- stride notes (should be empty) ---"
  $PY -c "
import glob, json
for f in sorted(glob.glob('$OUT/calib*/*/calibration.json')):
    c = json.load(open(f))
    print(f, '->', c.get('stride_notes') or 'none')
"
  echo "=== a_full done $(date -Is) ==="
} 2>&1 | tee "$LOG"
