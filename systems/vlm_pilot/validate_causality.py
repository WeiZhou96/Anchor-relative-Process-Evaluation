"""Causal boundary and artifact checks; no model or GPU needed."""
import hashlib,json,csv
from pathlib import Path
import numpy as np
from prepare import select_indices
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'systems/vlm_pilot'
def main():
    times=np.arange(101)/10
    early=select_indices(times,3.07)
    # Appending, removing or changing future timestamps must not affect the prefix.
    assert np.array_equal(early,select_indices(times[times<=3.07],3.07))
    altered=np.concatenate([times[times<=3.07],np.arange(1000,2000)])
    assert np.array_equal(early,select_indices(altered,3.07))
    assert max(times[early])<=3.07 and max(times[early])==3.0
    plan=json.loads((OUT/'prefix_audit.json').read_text())
    cfg=json.loads((OUT/'config.json').read_text())
    manifest=list(csv.DictReader((ROOT/'data/manifest/manifest_real.csv').open()))
    lookup={r['video_id']:r for r in manifest};ids={r['video_id'] for r in plan}
    assert len(plan)==220 and len(ids)==20
    assert hashlib.md5((ROOT/'data/manifest/manifest_real.csv').read_bytes()).hexdigest()==cfg['manifest_md5']
    for r in plan:
        assert lookup[r['video_id']]['split']=='dev'
        assert len(r['frame_times_s'])==8
        assert all(t<=r['end_s'] for t in r['frame_times_s'])
        assert all(a<b for a,b in zip(r['frame_times_s'],r['frame_times_s'][1:]))
    for vid in ids:assert sorted(r['offset_s'] for r in plan if r['video_id']==vid)==list(range(11))
    report={'status':'passed','queries':220,'clips':20,'future_timestamp_invariance':True,
        'future_frame_violations':0,'min_causal_margin_s':min(r['causal_margin_s'] for r in plan),
        'max_timestamp_approximation_s':max(r['timing_approximation_max_s'] for r in plan),
        'class_counts':{name:sum(lookup[v]['class_name']==name for v in ids) for name in ['head-on','rear-end','t-bone','sideswipe','single']},
        'unique_source_clusters':len({lookup[v]['source_cluster_id'] for v in ids})}
    (OUT/'causality_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
