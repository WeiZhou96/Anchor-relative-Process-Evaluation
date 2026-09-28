#!/usr/bin/env bash
# Read the metrics json already written by the fixed code and report the three
# cohort conventions plus the K5 granularity, for the horizon given as $1.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export PYTHONUNBUFFERED=1

H=${1:-10.03}
LOG=${APE_TMP}/a_verify_cohorts.log
{
  echo "=== a_verify_cohorts H=$H $(date -Is) ==="
  $PY -c "
import glob, json, os
H = float('$H')
target = None
def sysfiles(p):
    return sorted(f for f in os.listdir(p)
                  if f.endswith('.json') and not f.startswith('_'))
for p in sorted(glob.glob('outputs/metrics/*/')):
    fs = sysfiles(p)
    if not fs: continue
    j = json.load(open(os.path.join(p, fs[0])))
    if 'h_s' in j and abs(j['h_s'] - H) < 1e-9:
        target = p; break
if target is None:
    raise SystemExit('no metrics dir for H=%g' % H)
one = json.load(open(os.path.join(target, sysfiles(target)[0])))
print('dir:', target, ' H=', one['h_s'], ' effective_H=', one['effective_h_s'],
      ' N_H=', one['N_H'], ' n_systems=', len(sysfiles(target)))
print()
print('2. THREE COHORT CONVENTIONS (all three columns are now the same quantity)')
hdr = '%-28s %11s %11s %11s | %10s %10s %10s | %12s' % (
    'system','fixed_RMSCD','dyn_RMSCD','d2_RMSCD','fixed_S@1','dyn_S@1','d2_S@1','plainacc_S@1')
print(hdr); print('-'*len(hdr))
order = ['block__oracle__default','block__lock__d0-3','block__osc__d0-4_p-1',
         'block__rand__eta-0.9','block__rand__eta-0.5','block__frac__c-0.6']
for sid in order:
    f = os.path.join(target, sid + '.json')
    if not os.path.exists(f): continue
    a = json.load(open(f))['alt_cohorts']
    print('%-28s %11.4f %11.4f %11.4f | %10.4f %10.4f %10.4f | %12.4f' % (
        sid, a['fixed_H_cohort']['RMSCD'], a['dynamic_denominator']['RMSCD'],
        a['per_clip_end']['RMSCD'], a['fixed_H_cohort']['S_at_ref'],
        a['dynamic_denominator']['S_at_ref'], a['per_clip_end']['S_at_ref'],
        a['dynamic_denominator']['plain_accuracy_at_ref']))
print()
for sid in ['block__osc__d0-4_p-1','block__rand__eta-0.9']:
    f = os.path.join(target, sid + '.json')
    if not os.path.exists(f): continue
    a = json.load(open(f))['alt_cohorts']
    d = a['dynamic_denominator']
    print('%-24s plain_accuracy_area=%.4f  risk set %d -> %d  length_gap(d2=%.4f, fixed=%.4f)' % (
        sid, d['plain_accuracy_area'], d['n_at_start'], d['n_at_H'],
        a['length_stratified_change']['per_clip_end'],
        a['length_stratified_change']['fixed_H_cohort']))
print()
print('3. K5 GRANULARITY')
hdr = '%-28s %10s %12s %16s %18s' % ('system','K5','K5_video','K5_conf_cells','preanchor_conf_cells')
print(hdr); print('-'*len(hdr))
for sid in ['block__osc__d0-4_p-1','block__lock__d0-3','standin_early','standin_late',
            'standin_flippy','standin_smoothed']:
    f = os.path.join(target, sid + '.json')
    if not os.path.exists(f): continue
    p_ = json.load(open(f))['phenomena']
    print('%-28s %10s %12s %16s %18s' % (sid, p_['K5'], p_['K5_video_rate'],
          p_['K5_confident_cells'], p_['pre_anchor_confident_cells']))
f = os.path.join(target, 'block__osc__d0-4_p-1.json')
if os.path.exists(f):
    print()
    print('K5_definition:', json.load(open(f))['phenomena']['K5_definition'])
"
  echo
  echo "standin declared confidences vs protocol threshold:"
  $PY -c "
import yaml, glob, os
for d in sorted(glob.glob('outputs/answers_a_standin/*/system_card.yaml')):
    c = yaml.safe_load(open(d))
    print(' ', c['system_id'], 'confidence =', c['standin_params']['confidence'])
print('  protocol conf_threshold =', yaml.safe_load(open('protocol/pi0.yaml'))['phenomena']['conf_threshold'])
"
} 2>&1 | tee "$LOG"
