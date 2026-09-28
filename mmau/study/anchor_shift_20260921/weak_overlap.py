"""Apply the unchanged dense matcher to weak edges incident to the class-12/14 guard cases."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'overlap_dense_20260920'))
from dense_overlap import load_clip, match_pair, sha, test_chain


def main(project, root, out):
    old = project / 'deliverables/pixel_pilot_20260920/overlap'
    impact = json.loads((old / 'impact.json').read_text())
    ids = {r['video_id'] for item in impact['classes'].values() for r in item['videos']}
    pairs = [r for r in json.loads((old / 'pairs.json').read_text()) if r['evidence_class'] != 'strong_multiframe_similarity' and (r['a'] in ids or r['b'] in ids)]
    ledger_path = project / 'deliverables/route_feasibility_20260920/outputs/planning_ledger.jsonl'
    ledger = {r['hashcode']: r for r in map(json.loads, ledger_path.read_text().splitlines())}
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(selection='ALL prior weak edges incident to the 14 previously excluded class-12/14 dev H67 cases, including edges within their existing large guard component', pairs=[{'a':p['a'], 'b':p['b']} for p in pairs], focus_ids=sorted(ids), matcher_sha256=sha(Path(__file__).resolve().parent.parent / 'overlap_dense_20260920/dense_overlap.py'), source_sha256=sha(Path(__file__)), unchanged_thresholds=True, no_guard_relaxation=True, chain_test=test_chain())
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    unique = sorted({p[k] for p in pairs for k in ('a','b')})
    with ThreadPoolExecutor(max_workers=6) as pool:
        clips = dict(zip(unique, pool.map(lambda key: load_clip(ledger[key], root), unique)))
    (out / 'frames').mkdir()
    for key, clip in clips.items():
        (out / 'frames' / (key + '.json')).write_text(json.dumps({k:clip[k] for k in ('frames','raw_sha256','pixel_sha256','dhash')}))
    results = []
    for pair in pairs:
        a, b = pair['a'], pair['b']
        result = dict(a=a,b=b,released_partitions=[ledger[a]['released_partition'],ledger[b]['released_partition']], **match_pair(clips[a], clips[b], ledger[a], ledger[b]))
        results.append(result)
    (out / 'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    summary = dict(pairs=len(results), clips=len(unique), frames_read=sum(len(c['frames']) for c in clips.values()), status_counts=dict(Counter(r['status'] for r in results)), sample_exclusions_modified=False, source_identity_certified=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    main(a.project,a.root,a.out)
