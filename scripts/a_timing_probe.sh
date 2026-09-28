#!/usr/bin/env bash
# Validate a_full.sh and measure the real per-stage cost on the frozen protocol
# and the audit (test) split. Writes to a throwaway directory; outputs/ untouched.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export PYTHONUNBUFFERED=1

LOG=${APE_TMP}/a_timing_probe.log
TMP=${APE_TMP}/probe_out
rm -rf "$TMP"; mkdir -p "$TMP"

{
  echo "=== a_timing_probe $(date -Is) ==="

  echo "--- a_full.sh syntax ---"
  bash -n scripts/a_full.sh && echo "a_full.sh: syntax OK"

  echo "--- frozen protocol, audit cohorts, grid sizes ---"
  $PY -c "
from ape.protocol import load_protocol_checked, load_manifest, content_sha256
from ape.cohort import select_split, split_summary, cohort_table
cfg = load_protocol_checked('protocol/pi0.yaml')
print('version:', cfg.protocol_version, '| frozen:', cfg.frozen)
print('sha256 :', content_sha256('protocol/pi0.yaml'))
print('manifest_md5:', cfg.raw.get('manifest_md5'))
print('dev_hash    :', cfg.raw.get('dev_hash'))
print('audit_split :', cfg.raw.get('audit_split'))
for h in cfg.h_list_s:
    print('  pi_hash H=%-6g %s' % (h, cfg.pi0(h).pi_hash))
m = load_manifest('data/manifest/manifest_real.csv')
print('splits:', split_summary(m))
audit = select_split(m, 'test')
print('audit clips:', len(audit), '| clusters:', audit['source_cluster_id'].nunique())
print(cohort_table(audit, cfg.h_list_s).to_string())
print('grid: axis=%d, plaus-full=%d, full=%d' % (
    len(cfg.scan_grid(mode='axis')),
    len(cfg.scan_grid(plaus=True, mode='full')),
    len(cfg.scan_grid(mode='full'))))
"

  echo "--- timing: eval, ONE horizon, frozen bootstrap_n=1000, audit split ---"
  time $PY -m ape.cli eval --pi protocol/pi0.yaml \
      --manifest data/manifest/manifest_real.csv --split test \
      --answers outputs/answers --out "$TMP/metrics" --h 10.0

  echo "--- timing: axis scan (14 points) ---"
  time $PY -m ape.cli scan --pi protocol/pi0.yaml \
      --manifest data/manifest/manifest_real.csv --split test \
      --answers outputs/answers --out "$TMP/calib" --mode axis

  echo "--- timing: phenomena (one horizon) ---"
  time $PY -m ape.cli phenomena --pi protocol/pi0.yaml \
      --manifest data/manifest/manifest_real.csv --split test \
      --answers outputs/answers --out "$TMP/phenomena"

  echo "--- audit_split stamped into the outputs ---"
  $PY -c "
import glob, json, os
f = sorted(glob.glob('$TMP/metrics/*/block__lock__d0-3.json'))[0]
d = json.load(open(f))
print(os.path.basename(os.path.dirname(f)), d['system_id'],
      'audit_split=', d['audit_split'], 'N_H=', d['N_H'],
      'h_s=', d['h_s'], 'effective_h_s=', d['effective_h_s'], 'RMSCD=', d['RMSCD'])
c = json.load(open(sorted(glob.glob('$TMP/calib/*/calibration.json'))[0]))
print('calib audit_split=', c['audit_split'], 'n_clips_scored=', c['n_clips_scored'])
p = json.load(open(sorted(glob.glob('$TMP/metrics/*/_pairs.json'))[0]))
print('pairs audit_split=', p['audit_split'], 'n_real=', len(p['systems_real']))
"
  echo "=== a_timing_probe done $(date -Is) ==="
} 2>&1 | tee "$LOG"
