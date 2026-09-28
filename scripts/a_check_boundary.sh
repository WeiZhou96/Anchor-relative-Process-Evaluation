#!/usr/bin/env bash
# Find the clips that sit exactly on a horizon boundary, to explain the
# one-clip difference between the freeze artefact's N_H and ours.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export PYTHONUNBUFFERED=1

LOG=${APE_TMP}/a_check_boundary.log
{
  echo "=== a_check_boundary $(date -Is) ==="
  $PY -c "
import pandas as pd
df = pd.read_csv('data/manifest/manifest_real.csv')
t = df[df['split']=='test']
for h in (4.0, 10.0, 21.5):
    strict = int((t['post_anchor_length_s'] >= h).sum())
    tol    = int((t['post_anchor_length_s'] >= h - 1e-9).sum())
    print('H=%-5g strict>=H: %-5d  >=H-1e-9: %-5d  diff: %d' % (h, strict, tol, tol-strict))
    near = t[(t['post_anchor_length_s'] < h) & (t['post_anchor_length_s'] >= h - 1e-6)]
    for r in near.itertuples(index=False):
        print('   boundary clip:', r.video_id,
              'post=%r' % r.post_anchor_length_s,
              'duration=%r anchor=%r' % (r.duration_s, r.anchor_s),
              'duration-anchor=%r' % (r.duration_s - r.anchor_s))
print()
print('raw CSV text for those rows:')
import csv
want = set()
for h in (4.0, 10.0, 21.5):
    near = t[(t['post_anchor_length_s'] < h) & (t['post_anchor_length_s'] >= h - 1e-6)]
    want |= set(near['video_id'].astype(str))
with open('data/manifest/manifest_real.csv', newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
        if row['video_id'] in want:
            print('  ', row['video_id'], 'anchor_s=', row['anchor_s'],
                  'duration_s=', row['duration_s'],
                  'post_anchor_length_s=', row['post_anchor_length_s'])
"
} 2>&1 | tee "$LOG"
