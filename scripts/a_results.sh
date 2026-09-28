#!/usr/bin/env bash
# Summarise the full-run artefacts: cohorts, grid sizes, table 3 with per-metric
# rulers, pair counts, arm_rule coverage.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export PYTHONUNBUFFERED=1
LOG=${APE_TMP}/a_results.log
{
  echo "=== a_results $(date -Is) ==="
  $PY -c "
import glob, json, os
import pandas as pd

print('--- per-horizon eval ---')
for d in sorted(glob.glob('outputs/metrics/*/')):
    fs = [f for f in os.listdir(d) if f.endswith('.json') and not f.startswith('_')]
    if not fs: continue
    one = json.load(open(os.path.join(d, fs[0])))
    p = json.load(open(os.path.join(d, '_pairs.json')))
    print('pi_hash=%s H=%-5g eff_H=%-5g N_H=%-5d systems=%-3d real=%-3d blocks=%-3d split=%s'
          % (one['pi_hash'], one['h_s'], one['effective_h_s'], one['N_H'], len(fs),
             len(p['systems_real']), len(p['systems_block']), one['audit_split']))
    stray = [s for s in p['systems_real'] + p['systems_block'] if '__stride' in s]
    print('    stride dirs leaked into the system set:', stray or 'none')
    print('    window-end-tied pairs: %d | p_method: %s | legacy RMSCD ruler: %s'
          % (len(p['window_end_tied_pairs']), p.get('p_method'),
             p.get('ruler_rmscd_legacy')))
    for m in ['RMSCD@H','S_H@1','S_H@3','end_window_macro_acc','median_flips']:
        sig = p.get(m, {}).get('significant_pairs', [])
        print('      %-22s significant pairs: %-5d ruler_M=%s'
              % (m, len(sig), (p.get('rulers_by_metric') or {}).get(m)))

for label, root in (('axis','outputs/calib'), ('plaus-full','outputs/calib_plaus')):
    print()
    print('--- calibration: %s ---' % label)
    for f in sorted(glob.glob(os.path.join(root, '*', 'calibration.json'))):
        c = json.load(open(f))
        print('file:', f)
        print('  pi0=%s split=%s n_pi=%d n_systems=%d mode=%s' % (
            c['pi0_hash'], c.get('audit_split'), c['n_pi'], c['n_systems'],
            c.get('scan_mode')))
        print('  legacy RMSCD ruler: %s' % c.get('ruler_rmscd_legacy'))
        print('  stride re-runs used for %d systems; notes: %s'
              % (len(c.get('stride_reruns', {})), c.get('stride_notes') or 'none'))
        print('  %-22s %-9s %-11s %-11s %-9s %-9s %s'
              % ('metric','min_R','MRD_plaus','MRD_full','ruler_M','n_pairs','verdict'))
        for m in ['RMSCD@H','S_H@1','S_H@3','end_window_macro_acc','median_flips']:
            e = c['metrics'].get(m)
            if not e: continue
            r = e.get('ruler')
            print('  %-22s %-9.4g %-11.4g %-11.4g %-9s %-9s %s'
                  % (m, e['min_R_M_plaus'], e['MRD_plaus'], e['MRD_full_grid'],
                     ('%.4g' % r) if r is not None else 'None',
                     e['n_pairs'], e['verdict']))
print()
print('--- report index ---')
idx = 'outputs/report_index.json'
if os.path.exists(idx):
    d = json.load(open(idx))
    print('rows:', d['n_rows'], '->', os.path.abspath(idx))
    df = pd.read_csv('outputs/report_index.csv')
    print('distinct systems:', df['system_id'].nunique(),
          '| horizons:', sorted(df['h_s'].unique()))
    print('any __stride rows:', int(df['system_id'].astype(str).str.contains('__stride').sum()))
    print('max missing_lookup_rate:', float(df['missing_lookup_rate'].max()))
    print('arm_rule present:', 'arm_rule' in df.columns)
    if 'arm_rule' in df.columns:
        one_h = df[df['h_s'] == df['h_s'].min()]
        g = one_h.groupby('arm_rule')['seed'].nunique().sort_index()
        print('  arm_rule -> distinct seeds (at one horizon):')
        for k, v in g.items():
            n = int((one_h['arm_rule'] == k).sum())
            print('    %-18s seeds=%s systems=%d' % (k, v, n))
"
} 2>&1 | tee "$LOG"
