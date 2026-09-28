#!/usr/bin/env bash
# Show R_M along the common-mode shift axis, so that eps_max = 0 can be told
# apart from "R_M is undefined here".
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export PYTHONUNBUFFERED=1

LOG=${APE_TMP}/a_verify_eps.log
{
  echo "=== a_verify_eps $(date -Is) ==="
  $PY -c "
import glob, json, os
import pandas as pd
pd.set_option('display.width', 220)
d = sorted(glob.glob('outputs/calib/*/'))[0]
cal = json.load(open(os.path.join(d, 'calibration.json')))
print('dir:', d, ' pi0:', cal['pi0_hash'], ' r0:', cal['r0'])
print('n_pi:', cal['n_pi'], ' n_systems:', cal['n_systems'])
print('ruler (median progress gap among window-end-tied pairs):',
      cal['ruler_median_progress_diff_RMSCD'])
print()
scan = pd.read_csv(os.path.join(d, 'scan_table.csv'))
print('systems in scan:', scan['system_id'].nunique(),
      ' of which blocks:', scan[scan.is_block]['system_id'].nunique(),
      ' real:', scan[~scan.is_block]['system_id'].nunique())
print('real system ids:', sorted(scan[~scan.is_block]['system_id'].unique()))
print()
for m in ['RMSCD@H','end_window_macro_acc','median_flips']:
    f = os.path.join(d, 'R_' + m.replace('@','at') + '.csv')
    if not os.path.exists(f):
        print('missing', f); continue
    r = pd.read_csv(f)
    ax = r[(r.eps_jit_sd_s==0) & (r.delta_s==0.5) & (abs(r.h_s-10.03)<1e-9)]
    ax = ax.sort_values('eps_sys_s')
    print('---', m, '--- (n_pairs =', ax['n_pairs'].tolist(), ')')
    print(ax[['eps_sys_s','R_M','n_pairs','in_plaus']].to_string(index=False))
    e = cal['metrics'][m]['eps_max']
    print('eps_max:', e['eps_max'], ' scanned_max_abs_eps:', e['scanned_max_abs_eps'],
          ' n_points:', e['n_points'])
    print('MRD_plaus:', cal['metrics'][m]['MRD_plaus'],
          ' min_R_plaus:', cal['metrics'][m]['min_R_M_plaus'],
          ' verdict:', cal['metrics'][m]['verdict'])
    print()
"
} 2>&1 | tee "$LOG"
