#!/usr/bin/env bash
# Print the actual top-level key set of one metrics json and one calibration json,
# so the schema difference with the reporting stage can be arbitrated from facts.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

LOG=${APE_TMP}/a_verify_schema.log
{
  echo "=== a_verify_schema $(date -Is) ==="
  $PY -c "
import glob, json
p = sorted(glob.glob('outputs/metrics/*/block__lock__d0-3.json'))[0]
d = json.load(open(p))
print('file:', p)
print('top-level keys:', sorted(d))
print('alt_cohorts keys:', sorted(d['alt_cohorts']))
print('alt_cohorts.fixed_H_cohort:', d['alt_cohorts']['fixed_H_cohort'])
print('alt_cohorts.per_clip_end:', {k: v for k, v in d['alt_cohorts']['per_clip_end'].items() if k != 'note'})
print('alt_cohorts.length_stratified_change:', {k: v for k, v in d['alt_cohorts']['length_stratified_change'].items() if k != 'note'})
print('phenomena keys:', sorted(k for k in d['phenomena'] if k != 'full'))
print('commit keys:', sorted(d['commit']))
print('bootstrap keys:', sorted(d['bootstrap']))
print('track/family/effective_h_s:', d['track'], d['family'], d['effective_h_s'])
print()
c = sorted(glob.glob('outputs/calib/*/calibration.json'))[0]
cal = json.load(open(c))
print('calib file:', c)
print('calib top-level:', sorted(cal))
print('calib metric names:', sorted(cal['metrics']))
e = cal['metrics']['RMSCD@H']
print('calib RMSCD@H keys:', sorted(k for k in e if k != 'eps_max'))
print('calib RMSCD@H aliases:', {k: e[k] for k in ('reference_range','max_b','max_s','min_R','MRD','verdict')})
print('delta_star systems:', len(cal['delta_star']))
print('ruler:', cal['ruler_median_progress_diff_RMSCD'])
"
} 2>&1 | tee "$LOG"
