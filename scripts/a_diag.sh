#!/usr/bin/env bash
# Diagnose (1) whether the scan really covered the eps_sys axis and
# (2) what the three cohort conventions actually computed for two gauge blocks.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

LOG=${APE_TMP}/a_diag.log
{
  echo "=== a_diag $(date -Is) ==="
  $PY -c "
import glob, json
import pandas as pd
pd.set_option('display.width', 200)
c = sorted(glob.glob('outputs/calib/*/scan_table.csv'))[0]
s = pd.read_csv(c)
print('scan file:', c)
print('n pi:', s['pi_hash'].nunique(), ' n systems:', s['system_id'].nunique())
print('eps_sys_s values scanned:', sorted(s['eps_sys_s'].unique()))
print('eps_jit_sd_s values:', sorted(s['eps_jit_sd_s'].unique()))
print('delta_s values:', sorted(s['delta_s'].unique()))
print('h_s values:', sorted(s['h_s'].unique()))
print()
print('rows with jit==0 (the axis eps_max walks):')
z = s[(s.eps_jit_sd_s==0) & (s.metric=='RMSCD@H')][['pi_hash','eps_sys_s','delta_s','h_s']].drop_duplicates()
print(z.sort_values(['eps_sys_s','delta_s','h_s']).to_string(index=False))
print()
print('rows at eps==0 and jit==0 (what the old eps_max used to pick a reference from):')
print(z[z.eps_sys_s==0].to_string(index=False))
"
  echo "--- missing lookup rates (should be 0 everywhere) ---"
  $PY -c "
import glob, json
worst = 0.0
for p in glob.glob('outputs/metrics/*/*.json'):
    if p.split('/')[-1].startswith('_'): continue
    d = json.load(open(p))
    worst = max(worst, d.get('missing_lookup_rate') or 0.0)
print('max missing_lookup_rate across all metric files:', worst)
"
  echo "--- answer coverage: j_min / j_max per system ---"
  $PY -c "
import glob, os
import pandas as pd
for d in sorted(glob.glob('outputs/answers/block__lock__d0-3')) + sorted(glob.glob('outputs/answers_a_standin/*')):
    df = pd.read_csv(os.path.join(d,'answers.csv'), usecols=['j'])
    print(os.path.basename(d), 'j from', int(df.j.min()), 'to', int(df.j.max()))
"
} 2>&1 | tee "$LOG"
