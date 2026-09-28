#!/usr/bin/env bash
# Probe the server environment and track C's manifest.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

mkdir -p ${APE_TMP}
LOG=${APE_TMP}/a_probe.log
{
  echo "=== a_probe $(date -Is) ==="
  $PY -c "
import sys
print('python', sys.version)
for m in ['numpy','pandas','scipy','yaml','pytest','pyarrow']:
    try:
        mod = __import__(m)
        print('OK  ', m, getattr(mod,'__version__','?'))
    except Exception as e:
        print('MISS', m, type(e).__name__, e)
"
  echo "--- manifest ---"
  ls -l data/manifest/ 2>/dev/null || echo "no data/manifest yet"
  if [ -f data/manifest/manifest_real.csv ]; then
    $PY -c "
import pandas as pd
df = pd.read_csv('data/manifest/manifest_real.csv')
print('rows', len(df))
print('cols', list(df.columns))
print(df[['video_id','source_cluster_id','anchor_s','duration_s','post_anchor_length_s','class_code','split']].head(3).to_string())
print('clusters', df['source_cluster_id'].nunique())
print('splits', df['split'].value_counts().to_dict())
print('L+ quantiles', df['post_anchor_length_s'].quantile([.25,.5,.75]).round(3).to_dict())
for h in (4.11, 10.03, 21.82):
    print('eligible@H%g' % h, int((df['post_anchor_length_s'] >= h).sum()))
"
  fi
} 2>&1 | tee "$LOG"
