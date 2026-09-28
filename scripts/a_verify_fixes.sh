#!/usr/bin/env bash
# Verify the three fixes on the freshly re-run smoke outputs:
#   1. eps_max now finds the common-mode shift axis
#   2. the three cohort conventions are the same quantity for osc(1,4), rand(0.9)
#   3. K5 granularity and why the stand-ins report null
# Also confirm the pi_hash values did not move.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

LOG=${APE_TMP}/a_verify_fixes.log
{
  echo "=== a_verify_fixes $(date -Is) ==="

  echo "--- pi_hash of pi0 at each H (must be unchanged) ---"
  $PY -c "
from ape.protocol import load_protocol
cfg = load_protocol('protocol/pi0.yaml')
for h in cfg.h_list_s:
    pv = cfg.pi0(h)
    print('H=%-6g %s' % (h, pv.pi_hash))
"

  echo
  echo "--- 1. eps_max on every frozen metric (calibration.json) ---"
  $PY -c "
import glob, json
c = sorted(glob.glob('outputs/calib/*/calibration.json'))[0]
cal = json.load(open(c))
print('file:', c, ' pi0_hash:', cal['pi0_hash'])
for m in ['RMSCD@H','end_window_macro_acc','median_flips','S_H@1','S_H@3']:
    e = cal['metrics'][m]['eps_max']
    print('%-22s eps_max=%-6s scanned_max_abs_eps=%-6s n_points=%-3s ref(delta,H)=(%s,%s) saturated=%s'
          % (m, e['eps_max'], e['scanned_max_abs_eps'], e['n_points'],
             e.get('reference_delta_s'), e.get('reference_h_s'), e['saturated']))
"

  echo
  echo "--- 2. three cohort conventions, gauge blocks, H=10.03 (effective 10.0) ---"
  $PY -c "
import glob, json
d = sorted(glob.glob('outputs/metrics/*/'))
import os
# pick the H=10.03 directory
target = None
for p in d:
    fs = [f for f in os.listdir(p) if f.endswith('.json') and not f.startswith('_')]
    if not fs: continue
    j = json.load(open(os.path.join(p, fs[0])))
    if abs(j['h_s'] - 10.03) < 1e-9:
        target = p; break
print('dir:', target)
hdr = '%-26s %-10s %-10s %-10s %-10s %-10s %-10s' % ('system','fixed_RMSCD','dyn_RMSCD','d2_RMSCD','fixed_S@1','dyn_S@1','d2_S@1')
print(hdr); print('-'*len(hdr))
for sid in ['block__oracle__default','block__osc__d0-4_p-1','block__rand__eta-0.9','block__rand__eta-0.5','block__lock__d0-3']:
    j = json.load(open(os.path.join(target, sid + '.json')))
    a = j['alt_cohorts']
    print('%-26s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f' % (
        sid, a['fixed_H_cohort']['RMSCD'], a['dynamic_denominator']['RMSCD'], a['per_clip_end']['RMSCD'],
        a['fixed_H_cohort']['S_at_ref'], a['dynamic_denominator']['S_at_ref'], a['per_clip_end']['S_at_ref']))
print()
for sid in ['block__osc__d0-4_p-1','block__rand__eta-0.9']:
    j = json.load(open(os.path.join(target, sid + '.json')))
    a = j['alt_cohorts']['dynamic_denominator']
    print(sid, 'plain_accuracy_at_ref=%.4f plain_accuracy_area=%.4f risk n %d->%d'
          % (a['plain_accuracy_at_ref'], a['plain_accuracy_area'], a['n_at_start'], a['n_at_H']))
print()
for sid in ['block__osc__d0-4_p-1','block__rand__eta-0.9']:
    j = json.load(open(os.path.join(target, sid + '.json')))
    print(sid, 'length_stratified_change', {k: (round(v,4) if isinstance(v,float) else v)
          for k, v in j['alt_cohorts']['length_stratified_change'].items() if k != 'note'})
"

  echo
  echo "--- 3. K5 granularity: blocks vs stand-ins ---"
  $PY -c "
import glob, json, os
d = [p for p in sorted(glob.glob('outputs/metrics/*/')) ]
target = None
for p in d:
    fs = [f for f in os.listdir(p) if f.endswith('.json') and not f.startswith('_')]
    if not fs: continue
    j = json.load(open(os.path.join(p, fs[0])))
    if abs(j['h_s'] - 10.03) < 1e-9:
        target = p; break
hdr = '%-26s %-8s %-12s %-14s %-12s' % ('system','K5','K5_video','K5_conf_cells','preanchor_cells')
print(hdr); print('-'*len(hdr))
for sid in ['block__osc__d0-4_p-1','block__lock__d0-3','standin_early','standin_late','standin_flippy','standin_smoothed']:
    j = json.load(open(os.path.join(target, sid + '.json')))
    p_ = j['phenomena']
    print('%-26s %-8s %-12s %-14s %-12s' % (sid, p_['K5'], p_['K5_video_rate'],
          p_['K5_confident_cells'], p_['pre_anchor_confident_cells']))
print()
j = json.load(open(os.path.join(target, 'block__osc__d0-4_p-1.json')))
print('K5_definition:'); print(' ', j['phenomena']['K5_definition'])
print()
import yaml
for s in ['standin_early','standin_late','standin_flippy','standin_smoothed']:
    c = yaml.safe_load(open('outputs/answers_a_standin/%s/system_card.yaml' % s))
    print(s, 'declared confidence =', c['standin_params']['confidence'])
print('protocol conf_threshold =', yaml.safe_load(open('protocol/pi0.yaml'))['phenomena']['conf_threshold'])
"
} 2>&1 | tee "$LOG"
