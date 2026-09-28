#!/usr/bin/env bash
# Compare the realised system library against the frozen prereg system_library.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export PYTHONUNBUFFERED=1
LOG=${APE_TMP}/a_check_library.log
{
  echo "=== a_check_library $(date -Is) ==="
  $PY -c "
import collections, glob, os, re, yaml
suffix = '__stride'
dirs = [d for d in sorted(glob.glob('outputs/answers/*')) if os.path.isdir(d)
        and os.path.exists(os.path.join(d,'answers.csv'))]
base = [os.path.basename(d) for d in dirs if suffix not in os.path.basename(d)]
real = [b for b in base if not b.startswith('block__')]
print('base real systems:', len(real))

fam = collections.Counter()
arms = collections.defaultdict(set)
seeds = collections.Counter()
for s in real:
    if s.startswith('trivial__'): fam['trivial'] += 1
    elif s.startswith('clip__'):  fam['clip'] += 1
    elif s.startswith('prefix__'): fam['prefix'] += 1
    elif s.startswith('postproc__'):
        fam['postproc'] += 1; arms['postproc'].add(s.split('__')[1])
    elif s.startswith('commit__'):
        fam['commit'] += 1; arms['commit'].add(s.split('__')[1])
    else: fam['other'] += 1
    m = re.search(r'seed(\d+)', s)
    seeds[m.group(1) if m else 'none'] += 1
print('by family :', dict(fam))
print('seeds     :', dict(seeds))
print('postproc arms realised (%d):' % len(arms['postproc']), sorted(arms['postproc']))
print('commit arms realised   (%d):' % len(arms['commit']), sorted(arms['commit']))

pre = yaml.safe_load(open('prereg/S1_freeze_2026-09-03.yaml'))
lib = pre['system_library']
reg_pp = set(lib['postproc']['arms']); reg_cm = set(lib['commit']['arms'])
print()
print('prereg postproc arms (%d):' % len(reg_pp), sorted(reg_pp))
print('prereg commit arms   (%d):' % len(reg_cm), sorted(reg_cm))
print('prereg instance_count:', lib['instance_count'], '| realised:', len(real))
print()
print('postproc arms NOT in prereg :', sorted(arms['postproc'] - reg_pp))
print('prereg postproc arms MISSING:', sorted(reg_pp - arms['postproc']))
print('commit arms NOT in prereg   :', sorted(arms['commit'] - reg_cm))
print('prereg commit arms MISSING  :', sorted(reg_cm - arms['commit']))
"
} 2>&1 | tee "$LOG"
