#!/usr/bin/env python
"""Read-only audit of secondary artifacts and the saved primary baseline."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import hashlib,json,csv,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ape.r2 import read_json,dump,METRICS
from ape.r2c import Context,VLM,negative_control
from ape.r2b import revised_consequence
from ape.calib import rank_preservation

def verify(root=ROOT):
    out=Path(root)/'outputs';b=out/'r2b';c=out/'r2c';ctx=Context(root);count=0
    def check(ok,label):
        nonlocal count
        if not ok:raise AssertionError(label)
        count+=1
    check(len(ctx.ids)==185 and len(ctx.nontrivial)==169,'pool counts')
    check(ctx.meta[VLM]['seed'] is None,'VLM seed')
    check(ctx.table(VLM,.25) is not None,'VLM fine')
    for d in [.5,1.]:
        check(ctx.table(VLM,d).system_id==VLM,'stateless identity')
    index=read_json(c/'report_index.json')
    check(len(index['rows'])==555,'index 555')
    oldscan=pd.read_csv(b/'r2/scan_all.csv',float_precision='round_trip')
    newscan=pd.read_csv(c/'r2/scan_all.csv',float_precision='round_trip')
    cols=['system_id','pi_hash','metric'];rest=list(oldscan.columns)
    check(oldscan.sort_values(cols).reset_index(drop=True).equals(newscan[newscan.system_id!=VLM][rest].sort_values(cols).reset_index(drop=True)),'cached scalar equality')
    check(len(newscan)==185*96*5,'scan union cardinality')
    for h in ctx.cfg.h_list_s:
        ph=ctx.cfg.pi0(h).pi_hash
        recs={sid:read_json(c/'metrics'/ph/f'{sid}.json') for sid in ctx.ids}
        for sid in ctx.ids:
            if sid!=VLM:
                check((b/'metrics'/ph/f'{sid}.json').read_bytes()==(c/'metrics'/ph/f'{sid}.json').read_bytes(),'eval bytes '+sid)
        check(recs[VLM]['N_H']=={4.:1328,10.:1113,21.5:742}[h],'VLM cohort')
        check(recs[VLM]['commit']['rho']==0 and recs[VLM]['commit']['e_c'] is None,'no commitment')
        ps=read_json(c/'metrics'/ph/'_pairs.json')
        check(set(ps['systems_real'])==set(ctx.real),'P pool')
        for m in METRICS:
            pairs=ps[m]['significant_pairs']
            check(all(a in ctx.real and z in ctx.real and a!=z for a,z in pairs),'P identities')
            tied=ps['window_end_tied_pairs']
            ruler=float(np.median([abs(recs[a]['frozen_family'][m]-recs[z]['frozen_family'][m]) for a,z in tied]))
            check(np.isclose(ruler,ps['rulers_by_metric'][m],atol=1e-12,rtol=0),'independent ruler')
            for folder in ['calib_plaus','calib']:
                cal=read_json(c/folder/ph/'calibration.json')
                check(cal['metrics'][m]['n_pairs']==len(pairs),'calibration P')
                check(cal['n_pi']==(54 if folder=='calib_plaus' else 14),'frozen grid')
                scan=pd.read_csv(c/folder/ph/'scan_table.csv',float_precision='round_trip')
                actual=pd.read_csv(c/folder/ph/f'R_{m.replace("@","at")}.csv',float_precision='round_trip').set_index('pi_hash')
                expected=rank_preservation(scan,ph,m,pairs).set_index('pi_hash')
                check(np.allclose(actual.loc[expected.index].R_M,expected.R_M,rtol=0,atol=1e-12),'R exact from cached scan')
                check(float(actual.loc[ph].R_M)==1.,'reference R=1')
        oldboot=np.load(b/'r2'/f'bootstrap_{ph}.npz');newboot=np.load(c/'r2'/f'bootstrap_{ph}.npz')
        for key in oldboot.files:check(np.allclose(oldboot[key],newboot[key],rtol=0,atol=1e-12,equal_nan=True),'shared old bootstrap '+key)
    g3=read_json(c/'gates/g3_v5.json');history=read_json(b/'gates/g3.json')
    check(g3['primary']['original_rule']==history['original_rule'],'original primary history unchanged')
    check(g3['primary']['v4_23p5']==history['revised_rule'],'23.5 primary history unchanged')
    for pool,expected_n in [('primary',184),('secondary',185)]:
        d=g3[pool]['v5_22p0'];check(d['D2_visible_end_cap_s']==22.,'22 s record')
        for gh in d['by_h']:
            check(len(gh['rows'])==expected_n,'G3 rows')
            check(len(gh['oracle_negative_controls'])==2,'both oracles')
            for ctrl in gh['oracle_negative_controls']:
                check(ctrl['own_gap']==ctrl['fixed_gap']==0.,'oracle exact zero')
                check(ctrl['passed']==negative_control(ctrl['own_gap'],ctrl['fixed_gap'],ctrl['ruler']),'oracle rule')
            check(gh['computation_valid'],'G3 computation valid')
            for row in gh['rows']:
                check(row['coverage']['n_missing_visible']==0,'zero missing visible cells')
                check(row['consequence']==revised_consequence(row['own'],row['fixed'],row['MRD_plaus'],row['ruler']),'G3 row decision')
            owncounts=gh['rows'][0]['own']['counts']
            check(sum(owncounts)==gh['n_test_eligible'],'G3 same cohort')
            check(all(row['own']['counts']==row['fixed']['counts']==owncounts for row in gh['rows']),'paired strata')
    g4=read_json(c/'gates/g4.json')
    for gh in g4['by_h']:
        check(all(not row['same_seed'] for row in gh['pairs'] if VLM in [row['system_a'],row['system_b']]),'VLM not seeded')
        check(all(VLM not in [row['system_a'],row['system_b']] for g in gh['seed_groups'] for row in g['seeds']),'VLM cannot support seed replication')
    for p in (c/'tables').rglob('*.csv'):
        with p.open() as f:rows=list(csv.reader(f))
        check(bool(rows) and all(len(r)==len(rows[0]) for r in rows),'table width '+p.name)
    for tag in ['','_H4p00','_H10p00','_H21p50']:
        with (c/'tables'/('table1_audit'+tag+'.csv')).open() as f:rows=list(csv.DictReader(f))
        check(len(rows)==185 and sum(r['System']==VLM and r['train_data_unknown']=='True' for r in rows)==1,'table 1 VLM')
        with (c/'tables'/('table1b_audit_by_arm_rule'+tag+'.csv')).open() as f:rows=list(csv.DictReader(f))
        check(len(rows)==57 and sum(r['Seeds']=='n/a (deterministic)' for r in rows)==1,'table 1b seed discipline')
    check(len(list((c/'figs').glob('fig*.png')))==8,'eight PNG figures')
    check(len(list((c/'figs').glob('fig*.pdf')))==8,'eight vector PDF figures')
    result=dict(status='passed',assertions=count,old_eval_files_unchanged=552,scan_union_protocols=96,
        secondary_systems=185,primary_systems=184,zero_g3_visible_missing=True,
        all_oracle_gaps_zero=True,primary_history_preserved=True)
    dump(result,c/'verification.json');print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':verify()
