"""Summarize candidate evidence without modifying guard decisions."""
import json
from collections import Counter
from pathlib import Path

base = Path(__file__).resolve().parent
ledger = {r['hashcode']: r for r in map(json.loads, (base.parent / 'route_feasibility_20260920/outputs/planning_ledger.jsonl').read_text(encoding='utf-8').splitlines())}
components = json.loads((base.parent / 'native_task_20260920/outputs/development_verified/guard_components.json').read_text())
groups = {v: c for c in components for v in c['members']}
pairs = json.loads((base / 'overlap/pairs.json').read_text())
cross = [p for p in pairs if ledger[p['a']]['released_partition'] != ledger[p['b']]['released_partition']]
summary = {'cross_partition_candidate_edges': len(cross), 'cross_partition_evidence': dict(Counter(p['evidence_class'] for p in cross)), 'classes': {}}
for cls in (12, 14):
    impacted = [r for r in ledger.values() if r['source'] == 'cap' and r['released_partition'] == 'val' and r['planning_candidate'] and r['post_anchor_frames'] >= 67 and r['native_class'] == cls and groups[r['hashcode']]['quarantine']]
    items = []
    for row in impacted:
        key = row['hashcode']
        incident = [p for p in pairs if key in (p['a'], p['b'])]
        items.append({'video_id': key, 'component_size': len(groups[key]['members']), 'direct_edges': len(incident), 'strong_edges': sum(p['evidence_class'] == 'strong_multiframe_similarity' for p in incident), 'strong_cross_partition_edges': sum(p['evidence_class'] == 'strong_multiframe_similarity' and ledger[p['a']]['released_partition'] != ledger[p['b']]['released_partition'] for p in incident)})
    summary['classes'][str(cls)] = {'dev_h67_removed_by_guard': len(items), 'videos': items}
summary['decision'] = 'Retain conservative guard; weak probe evidence does not justify reintroduction.'
(base / 'overlap/impact.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
