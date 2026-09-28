"""Summarize all planned output-grid cells without picking favorable seeds."""
import itertools
import json
import statistics
from pathlib import Path

base = Path(__file__).resolve().parent / 'sensitivity'
runs = json.loads((base / 'results.json').read_text())['runs']
def arm(system):
    if system == 'train_majority':
        return system
    name = system.rsplit('_', 1)[0]
    return {'gru': 'gru_weighted', 'prefix_mean': 'mean_weighted'}.get(name, name)
arms = sorted({arm(r['system']) for r in runs})
cells = []
for name, h, d in itertools.product(arms, [32, 48, 64], [1, 2, 4, 8]):
    rr = [r for r in runs if arm(r['system']) == name and r['horizon'] == h and r['delta'] == d]
    cells.append(dict(arm=name, horizon=h, delta=d, metrics={k: statistics.mean(r['metrics'][k] for r in rr) for k in rr[0]['metrics']}))
lookup = {(r['arm'], r['horizon'], r['delta']): r['metrics'] for r in cells}
changes, reversals = [], []
key = 'RMSCD@H_norm_macro'
for name, h in itertools.product(arms, [32, 48, 64]):
    a, b = lookup[name, h, 1], lookup[name, h, 8]
    changes.append(dict(arm=name, horizon=h, macro_rmscd_delta8_minus_delta1=b[key] - a[key], flips_delta1=a['mean_flips'], flips_delta8=b['mean_flips']))
    assert b['mean_flips'] <= a['mean_flips'] + 1e-12
for h, d, pair in itertools.product([32, 48, 64], [2, 4, 8], list(itertools.combinations(arms, 2))):
    a, b = pair
    initial = lookup[a, h, 1][key] - lookup[b, h, 1][key]
    changed = lookup[a, h, d][key] - lookup[b, h, d][key]
    if initial * changed < -1e-12:
        reversals.append(dict(horizon=h, delta=d, arms=[a, b], difference_at_delta1=initial, difference_at_delta=changed))
horizon_reversals = []
for a, b in itertools.combinations(arms, 2):
    short = lookup[a, 32, 1][key] - lookup[b, 32, 1][key]
    long = lookup[a, 64, 1][key] - lookup[b, 64, 1][key]
    if short * long < -1e-12:
        horizon_reversals.append(dict(arms=[a, b], difference_h32=short, difference_h64=long))
report = dict(cells=cells, delta8_vs1=changes, macro_rmscd_pair_reversal_cells=reversals, horizon32_to64_pair_reversals=horizon_reversals,
              paired_seed_uncertainty_not_estimated=True, source_certification=False)
(base / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'max_grid_change': max(changes, key=lambda r: abs(r['macro_rmscd_delta8_minus_delta1'])), 'grid_reversals': reversals, 'horizon_reversals': horizon_reversals}, indent=2))
for name in arms:
    print(name, [(h, round(lookup[name,h,1][key],4), round(lookup[name,h,8][key],4)) for h in (32,48,64)])
