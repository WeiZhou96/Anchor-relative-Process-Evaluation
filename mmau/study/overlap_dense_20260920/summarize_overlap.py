"""Join correspondence evidence to metadata without changing any exclusions."""
from collections import Counter
import json
import statistics
from pathlib import Path

base = Path(__file__).resolve().parent
ledger = {r['hashcode']: r for r in map(json.loads, (base.parent / 'route_feasibility_20260920/outputs/planning_ledger.jsonl').read_text(encoding='utf-8').splitlines())}
pairs = [json.loads(line) for line in (base / 'dense/pairs.jsonl').read_text().splitlines()]
prior = json.loads((base.parent / 'pixel_pilot_20260920/overlap/impact.json').read_text())
supported = [p for p in pairs if p['status'] == 'dense_visual_overlap_supported']
segments = json.loads((base / 'dense/correspondence_runs.json').read_text())
ratios = [max(r['matched_span_frame_ratio_a_over_b'], 1 / r['matched_span_frame_ratio_a_over_b']) for r in segments]
known = json.loads((base.parent / 'continuation_20260920/outputs/content_constraints.json').read_text())['components']
known_cross = {tuple(sorted(c['members'])) for c in known if len(c['official_splits']) > 1}
found_exact = {tuple(sorted([p['a'], p['b']])) for p in pairs if p['all_frames_same_bytes']}
assert found_exact == known_cross
focus = {}
for cls, item in prior['classes'].items():
    impacted = {v['video_id'] for v in item['videos']}
    hits = [p for p in pairs if p['a'] in impacted or p['b'] in impacted]
    focus[cls] = [dict(a=p['a'], b=p['b'], status=p['status'], coverage=p['shorter_clip_coverage'], native_classes=p['native_classes']) for p in hits]
summary = dict(partition_pairs=dict(Counter('/'.join(sorted(p['released_partitions'])) for p in pairs)),
               native_label_differences=[dict(a=p['a'], b=p['b'], native_classes=p['native_classes'], coverage=p['shorter_clip_coverage']) for p in supported if p['native_labels_differ']],
               insufficient=[dict(a=p['a'], b=p['b'], matches=p['ordered_matches'], coverage=p['shorter_clip_coverage'], distinct=p['distinct_matched_dhashes']) for p in pairs if p['status'] != 'dense_visual_overlap_supported'],
               full_byte_pairs=[dict(a=p['a'], b=p['b']) for p in pairs if p['all_frames_same_bytes']],
               class12_14_known_guard_impact=focus,
               source_pairs=dict(Counter('/'.join(sorted([ledger[p[k]]['source'] for k in ('a','b')])) for p in pairs)),
               formal_source_identity_certified=False, sample_exclusions_modified=False)
summary['symmetric_frame_span_ratio_median'] = statistics.median(ratios)
summary['shorter_coverage_min_median'] = [min(p['shorter_clip_coverage'] for p in pairs), statistics.median(p['shorter_clip_coverage'] for p in pairs)]
summary['exact_pairs_equal_previous_five_cross_partition_pairs'] = True
(base / 'dense/metadata_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ('partition_pairs', 'source_pairs', 'symmetric_frame_span_ratio_median', 'shorter_coverage_min_median', 'exact_pairs_equal_previous_five_cross_partition_pairs')}, indent=2))
