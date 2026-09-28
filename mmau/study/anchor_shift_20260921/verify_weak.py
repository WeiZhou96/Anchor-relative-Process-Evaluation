"""Audit the focused weak-candidate evidence and relate it to prior guard cases."""
import json
from pathlib import Path

base=Path(__file__).resolve().parent
rows=[json.loads(line) for line in (base/'weak/pairs.jsonl').read_text().splitlines()]
prior=json.loads((base.parent/'pixel_pilot_20260920/overlap/impact.json').read_text())
checks=[]
for row in rows:
    a,b=[json.loads((base/'weak/frames'/f'{row[k]}.json').read_text()) for k in ('a','b')]
    last=(-1,-1)
    for m in row['correspondences']:
        assert m['i']>last[0] and m['j']>last[1]
        assert a['frames'][m['i']]==m['frame_a'] and b['frames'][m['j']]==m['frame_b']
        assert m['mae']<=.02 and m['correlation']>=.98
        last=m['i'],m['j']
    checks.append(dict(a=row['a'],b=row['b'],status=row['status'],matches=row['ordered_matches'],coverage=row['shorter_clip_coverage'],classes=row['native_classes']))
focus={}
for cls,info in prior['classes'].items():
    focus[cls]=[]
    for video in info['videos']:
        incident=[r for r in rows if video['video_id'] in (r['a'],r['b'])]
        focus[cls].append(dict(video_id=video['video_id'],prior_strong_edges=video['strong_edges'],new_supported_edges=sum(r['status']=='dense_visual_overlap_supported' for r in incident)))
report=dict(all_checks_passed=True,pairs=checks,focus=focus,newly_supported_previously_weak_only_videos=sum(v['prior_strong_edges']==0 and v['new_supported_edges']>0 for values in focus.values() for v in values),guard_unchanged=True)
(base/'weak/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
