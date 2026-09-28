"""Validate K-a outputs against frozen inputs and paired-statistic invariants."""
from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import pandas as pd
from ape.r2 import Context,read_json,dump,METRICS
from ape.protocol import verify_frozen

ctx=Context(ROOT);O=ctx.out
verify_frozen(ctx.cfg)
before=ROOT.parent/'frozen_before.json'
unchanged={}
if before.exists():
    for rel,sha in read_json(before).items():
        unchanged[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha
    assert all(unchanged.values()),unchanged
dev_sha=hashlib.sha256('\n'.join(sorted(ctx.dev.video_id)).encode()).hexdigest()
assert dev_sha=='31ed09e84d2c12ffa37786ed1810b9d4932bb8063ee36d1497cff48252025565'
assert hashlib.md5((ROOT/'data/manifest/manifest_real.csv').read_bytes()).hexdigest()=='b7a976594deeaed8cfff0ea1630dbd96'
for h in ctx.cfg.h_list_s:
    ref=ctx.cfg.pi0(h)
    for folder,points in [('calib_plaus',54),('calib',14)]:
        path=O/folder/ref.pi_hash
        c=read_json(path/'calibration.json');scan=pd.read_csv(path/'scan_table.csv')
        assert c['n_pi']==points and c['n_systems']==64
        assert len(scan)==points*64*5 and scan.pi_hash.nunique()==points
        assert scan.system_id.nunique()==64 and not scan.system_id.str.contains('__stride').any()
        assert c['reference_stride_reruns_applied']
        for metric in METRICS:
            rt=pd.read_csv(path/f'R_{metric.replace("@","at")}.csv')
            r=rt.loc[rt.pi_hash==ref.pi_hash,'R_M'].iloc[0]
            assert r==1,(h,metric,r)
            assert 0<=c['metrics'][metric]['min_R']<=1
g4=read_json(O/'gates/g4.json')
for h in g4['by_h']:
    boot=np.load(O/'r2'/f"bootstrap_{h['pi_hash']}.npz")
    for r in h['pairs']:
        for side in ['a','b']:
            sid=r[f'system_{side}']
            r[f'end_window_ci_{side}']=np.quantile(boot[f'{sid}::end_window_macro_acc'],[.025,.975]).tolist()
        lo=max(r['end_window_ci_a'][0],r['end_window_ci_b'][0])
        hi=min(r['end_window_ci_a'][1],r['end_window_ci_b'][1])
        assert lo<=hi
        if r['significant']:
            assert r['p_adjusted']<=.05
            assert r['ci_holm_lo']>0 or r['ci_holm_hi']<0
            assert r['ci_lo']>0 or r['ci_hi']<0
dump(g4,O/'gates/g4.json')
for name in ['g3','g4','g5']:
    assert (O/'gates'/f'{name}.json').exists()
for name in ['a1','a3','a4','a5']:
    assert (O/'ablations'/f'{name}.json').exists()
for name in ['p_b','p_c']:
    assert (O/'mechanisms'/f'{name}.json').exists()
scan=pd.read_csv(O/'r2/scan_all.csv').drop_duplicates(['system_id','pi_hash'])
plaus=pd.read_csv(O/'calib_plaus'/ctx.cfg.pi0().pi_hash/'scan_table.csv').drop_duplicates(['system_id','pi_hash'])
integrity=dict(frozen_unchanged=unchanged,dev_sha256=dev_sha,all_reference_R_equal_one=True,
               n_union_pi=int(scan.pi_hash.nunique()),n_missing_cells_union=int(scan.n_missing_lookup.sum()),
               n_system_protocol_cells_with_missing=int((scan.n_missing_lookup>0).sum()),
               max_missing_lookup_rate_union=float(scan.missing_lookup_rate.max()),
               n_missing_cells_plaus=int(plaus.n_missing_lookup.sum()),
               max_missing_lookup_rate_plaus=float(plaus.missing_lookup_rate.max()))
dump(integrity,O/'r2/verification.json')
print(json.dumps(integrity,indent=2))
