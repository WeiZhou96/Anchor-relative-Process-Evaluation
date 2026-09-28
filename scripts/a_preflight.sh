#!/usr/bin/env bash
# Confirm the stride discovery logic classifies B's 130 directories correctly
# BEFORE committing to the full run.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export PYTHONUNBUFFERED=1

LOG=${APE_TMP}/a_preflight.log
{
  echo "=== a_preflight $(date -Is) ==="
  $PY -c "
from ape.cli import _load_answer_tables, _split_stride_tables, _needs_rerun_per_step, _table_for_step, _is_block
from ape.protocol import load_protocol_checked

cfg = load_protocol_checked('protocol/pi0.yaml')
suffix = str(cfg.answers_cfg.get('stride_suffix', '__stride'))
raw = _load_answer_tables('outputs/answers')
print('directories discovered      :', len(raw))
base, strides = _split_stride_tables(raw, suffix)
blocks = [t for t in base if _is_block(t)]
real   = [t for t in base if not _is_block(t)]
print('base systems after split    :', len(base), '(blocks %d, real %d)' % (len(blocks), len(real)))
print('systems with stride re-runs :', len(strides))
n_stride = sum(len(v) for v in strides.values())
print('stride tables held aside    :', n_stride)
print('accounting: %d base + %d stride = %d (expect %d)' % (len(base), n_stride, len(base)+n_stride, len(raw)))

orphan = [t.system_id for t in base if suffix in t.system_id]
print('orphan stride dirs treated as systems (must be empty):', orphan)

need = sorted(t.system_id for t in real if _needs_rerun_per_step(t))
print('declare subsampling_equivalent=false:', len(need))
missing = [(s, f) for s in need for f in (2, 4) if f not in strides.get(s, {})]
print('missing re-runs (must be empty):', missing[:10], '...' if len(missing) > 10 else '')

# the stride tables must carry the coarse step, and be read under the base id
import collections
steps = collections.Counter()
for sid, d in strides.items():
    for f, t in d.items():
        steps[(f, t.delta_s)] += 1
print('stride (factor, delta_s) counts:', dict(steps))

notes = []
for pv_delta in (0.25, 0.5, 1.0):
    ids = set()
    srcs = collections.Counter()
    for t in base:
        u, _ = _table_for_step(t, strides, pv_delta, cfg.delta_fine_s, notes)
        ids.add(u.system_id)
        srcs[u.card.get('answers_read_from', 'own')] += 1
    extra = {i for i in ids if suffix in i}
    print('delta=%-5g scored ids=%d  stride-sourced=%d  ids containing %s: %s'
          % (pv_delta, len(ids), sum(v for k, v in srcs.items() if k != 'own'), suffix, extra or 'none'))
print('fallback notes (must be empty):', notes[:5])
"
  echo "=== a_preflight done $(date -Is) ==="
} 2>&1 | tee "$LOG"
