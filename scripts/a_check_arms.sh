#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}
export PYTHONUNBUFFERED=1
LOG=${APE_TMP}/a_check_arms.log
{
  echo "=== a_check_arms $(date -Is) ==="
  $PY -c "
import collections, glob, os, re
dirs = [os.path.basename(d) for d in sorted(glob.glob('outputs/answers/*'))
        if os.path.isdir(d) and '__stride' not in os.path.basename(d)]
grid = collections.defaultdict(set)
for s in dirs:
    if not s.startswith(('postproc__','commit__','prefix__','clip__')): continue
    arm = s.split('__')[1]
    m = re.search(r'seed(\d+)', s)
    grid[(s.split('__')[0], arm)].add(m.group(1) if m else 'none')
print('%-10s %-16s %s' % ('family','arm','seeds'))
for (fam, arm), seeds in sorted(grid.items()):
    flag = '' if len(seeds) == 3 else '   <-- NOT all three seeds'
    print('%-10s %-16s %s%s' % (fam, arm, sorted(seeds), flag))
print()
n_full = sum(1 for v in grid.values() if len(v) == 3)
print('arms with all 3 seeds: %d of %d' % (n_full, len(grid)))
"
} 2>&1 | tee "$LOG"
