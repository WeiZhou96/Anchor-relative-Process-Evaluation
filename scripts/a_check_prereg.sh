#!/usr/bin/env bash
# Compare protocol/pi0.yaml against C's S1 freeze artefact, and inventory the
# v2 manifest splits. The freeze artefact wins on any disagreement.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export PYTHONUNBUFFERED=1

LOG=${APE_TMP}/a_check_prereg.log
{
  echo "=== a_check_prereg $(date -Is) ==="
  echo "--- prereg dir ---"
  ls -l prereg/ 2>/dev/null || echo "no prereg dir"
  echo
  echo "--- C freeze artefact ---"
  cat prereg/S1_freeze_2026-09-03.yaml 2>/dev/null || echo "MISSING"
  echo
  echo "--- manifest v2 inventory ---"
  $PY -c "
import hashlib
import pandas as pd
df = pd.read_csv('data/manifest/manifest_real.csv')
print('rows:', len(df))
print('splits:', df['split'].value_counts().to_dict())
raw = open('data/manifest/manifest_real.csv','rb').read()
print('manifest md5:', hashlib.md5(raw).hexdigest())
dev = sorted(df.loc[df['split']=='dev','video_id'].astype(str))
payload = '\n'.join(dev).encode('utf-8')
print('dev clips:', len(dev), 'dev_hash sha256:', hashlib.sha256(payload).hexdigest())
test = df.loc[df['split']=='test']
print('test rows:', len(test))
for h in (4.0, 10.0, 21.5):
    print('  eligible@H%-5g all=%-5d test=%-5d' % (
        h, int((df['post_anchor_length_s']>=h).sum()),
        int((test['post_anchor_length_s']>=h).sum())))
print('test clusters:', test['source_cluster_id'].nunique())
"
  echo
  echo "--- clusters with more than one collision type ---"
  $PY -c "
import pandas as pd
df = pd.read_csv('data/manifest/manifest_real.csv')
g = df.groupby('source_cluster_id')['class_code'].nunique()
print('multi-type clusters (all splits):', int((g>1).sum()))
t = df[df['split']=='test']
gt = t.groupby('source_cluster_id')['class_code'].nunique()
print('multi-type clusters (test only) :', int((gt>1).sum()))
"
} 2>&1 | tee "$LOG"
