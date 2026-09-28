"""K-c: validated VLM secondary pool and the frozen v5 G3 record.

Primary calibration remains in r2b. All writes are under r2c.
"""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from . import r2, r2b
from .r2b_identity import metadata, group_key
from .cli import _intervals_from_samples, _phenomena_block, _track_of
from .metrics import alt_cohort_report
from .cohort import cluster_codes
from .protocol import TOL

VLM = 'r2__qwen25vl7b__readonly'
CAP = 22.0


class Context(r2.Context):
    def __init__(self, root, include_vlm=True):
        self.include_vlm=include_vlm
        super().__init__(root, Path(root)/('outputs/r2c' if include_vlm else 'outputs/r2b'),
                         (185,173,169) if include_vlm else (184,172,168))
        self.meta={s:metadata(s,self.cards) for s in self.ids}
    def provenance(self):
        return dict(super().provenance(),methods='REPORT_R2_Kc.md',freeze_version='v6',
                    g3_rule_version='v5 A5-1',vlm_included=self.include_vlm)


def prepare(root):
    root=Path(root);out=root/'outputs/r2c';out.mkdir(exist_ok=True)
    old=root/'outputs/r2b'
    if (out/'report_index.json').exists():
        raise FileExistsError('K-c already prepared; resume a named stage instead')
    index=r2.read_json(old/'report_index.json')
    rows=copy.deepcopy(index['rows'])
    assert len(rows)==552 and all(r['system_id']!=VLM for r in rows)
    import yaml
    card=yaml.safe_load((root/'outputs/answers'/VLM/'system_card.yaml').read_text())
    assert card['status']=='complete' and card['train_data_unknown'] is True
    assert card['stateless'] is True and card['subsampling_equivalent'] is True
    assert not card['stride_directories'] and 'seed' not in card
    meta=metadata(VLM,{VLM:card})
    for h,ph in [(4.,'77200b351bf1'),(10.,'8ac32aae418b'),(21.5,'3fb251dc210c')]:
        rows.append(dict(system_id=VLM,h_s=h,pi_hash=ph,**meta))
    index.update(n_rows=len(rows),rows=rows,metrics_root='r2c/metrics',outputs_root='r2c',
                 primary_outputs_root='r2b',secondary_vlm=True)
    r2.dump(index,out/'report_index.json')
    oldlib=r2.read_json(old/'library.json');oldlib.update(n_systems=185,n_vlm=1,
        system_ids=sorted(oldlib['system_ids']+[VLM]),primary_systems=184,vlm_seed=None)
    r2.dump(oldlib,out/'library.json')
    print('[prepare] 185 systems; 284 stand-ins; VLM has no seed or stand-ins',flush=True)


def evaluate(ctx):
    """Reuse immutable per-system K-b records; evaluate the new applicant only."""
    for h in ctx.cfg.h_list_s:
        ref=ctx.cfg.pi0(h)
        dest=ctx.out/'metrics'/ref.pi_hash;dest.mkdir(parents=True,exist_ok=True)
        for sid in ctx.ids:
            if sid!=VLM:
                shutil.copy2(ctx.root/'outputs/r2b/metrics'/ref.pi_hash/f'{sid}.json',dest/f'{sid}.json')
        res=ctx.eval(VLM,ref)
        reps,w=r2.bootstrap_weights(cluster_codes(pd.DataFrame({'source_cluster_id':res.source_cluster_id})),ctx.cfg.bootstrap_n,ctx.cfg.seed)
        points=res.frozen_family(ctx.cfg.s_report_delta_s)
        samples=r2.family_bootstrap(res,reps,w)
        payload=res.to_json_dict(ctx.cfg.s_report_delta_s)
        payload.update(ctx.meta[VLM]);payload.update(audit_split='test',n_clusters=len(np.unique(res.source_cluster_id)),
            track=_track_of(ctx.test),family=ctx.cards[VLM]['family'],group_key=group_key(ctx.meta[VLM]))
        payload['bootstrap']=_intervals_from_samples(points,samples,ctx.cfg.alpha_pairs,len(reps))
        payload['commit']['tau_c']=payload['commit']['tau_c_mean']
        table=ctx.table(VLM,ref.delta_s)
        payload['alt_cohorts']=alt_cohort_report(ctx.test,table,ref,ctx.cfg.grid_max_s,ctx.cfg.s_report_delta_s[0])
        payload['phenomena']=_phenomena_block(res,ctx.test,table,ctx.cfg)
        r2.dump(payload,dest/f'{VLM}.json')
        print(f'[eval VLM] H={h} {points}',flush=True)


def calibrate(ctx):
    """Reuse per-system scan values, recompute every pooled quantity with the VLM."""
    src=ctx.root/'outputs/r2b/r2/scan_all.csv'
    old=pd.read_csv(src,float_precision='round_trip')
    assert set(old.system_id)==set(ctx.ids)-{VLM}
    grid=[]
    for ph,group in old.groupby('pi_hash',sort=False):
        row=group.iloc[0]
        pv=ctx.cfg.pi0(float(row.h_s)).replace(**{k:float(row[k]) for k in r2.KNOBS})
        assert pv.pi_hash==ph
        grid.append(pv)
    single=copy.copy(ctx);single.ids=[VLM]
    new=r2.make_scan(single,grid)
    scan=pd.concat([old,new],ignore_index=True)
    r2.calibrate(ctx,scan=scan)
    ctx.save('cache_reuse',dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        old_systems=184,new_systems=1,n_protocols=len(grid),old_scalar_values=len(old),new_scalar_values=len(new),
        pooled_quantities_recomputed=['P','ruler','R_M','MRD','b_M','s_M','eps_max','delta_star'],
        policy='Unchanged per-system eval and scan values are reused; all with-VLM pooled statistics are newly computed.'),folder='r2')


def negative_control(own_gap, fixed_gap, ruler):
    values=[own_gap,fixed_gap,ruler]
    return bool(all(v is not None and np.isfinite(v) for v in values) and ruler>=0
                and abs(own_gap)<=ruler and abs(fixed_gap)<=ruler)


def checked_gate(groups, controls):
    valid=bool(controls) and all(c['passed'] for c in controls)
    if not valid:
        return dict(computation_valid=False,status='invalid_computation',pass_gate=None)
    available=all(groups[k]['n_undefined']==0 for k in ['random_block','real'])
    if not available:
        return dict(computation_valid=True,status='undefined',pass_gate=None)
    passed=groups['random_block']['n_consequence']==2 and groups['real']['n_consequence']>=.5*groups['real']['n']
    return dict(computation_valid=True,status='passed' if passed else 'failed',pass_gate=passed)


def corrected_g3(ctx):
    by=[]
    for h in ctx.cfg.h_list_s:
        ref=ctx.cfg.pi0(h)
        cohort=ctx.test[ctx.test.post_anchor_length_s>=h-TOL].reset_index(drop=True)
        dev=ctx.dev[ctx.dev.post_anchor_length_s>=h-TOL]
        cuts=np.quantile(dev.post_anchor_length_s,[1/3,2/3]);lengths=cohort.post_anchor_length_s.to_numpy()
        _,w=r2.bootstrap_weights(cluster_codes(cohort),ctx.cfg.bootstrap_n,ctx.cfg.seed)
        cal=r2.read_json(ctx.out/'calib_plaus'/ref.pi_hash/'calibration.json')
        mrd=cal['metrics']['RMSCD@H']['MRD_plaus'];ruler=cal['rulers_by_metric']['RMSCD@H']
        rows=[];controls=[]
        for sid in ctx.ids:
            res=ctx.eval(sid,ref);assert np.array_equal(res.video_ids,cohort.video_id)
            ownv,coverage=r2b.own_visible_delays(cohort,ctx.table(sid,ref.delta_s),ref.delta_s,cap=CAP)
            own=r2b.bootstrap_gap(ownv,lengths,cuts,w)
            fixed=r2b.bootstrap_gap(r2.rmscd_per_video(res.indicator,res.delta_s),lengths,cuts,w)
            card=ctx.cards[sid]
            role='random_block' if card.get('block_family')=='rand' else 'other_block' if sid in ctx.blocks else 'real' if sid in ctx.nontrivial else 'trivial'
            row=dict(system_id=sid,role=role,own=own,fixed=fixed,MRD_plaus=mrd,ruler=ruler,
                consequence=r2b.revised_consequence(own,fixed,mrd,ruler),coverage=coverage)
            rows.append(row)
            # v5 says oracle block; also require the card-declared ground-truth
            # trivial anchor, which is the actual 23.5-s cache-failure sentinel.
            if card.get('block_family')=='oracle' or card.get('uses_ground_truth') is True:
                og=own['gap_long_minus_short'];fg=fixed['gap_long_minus_short']
                controls.append(dict(system_id=sid,own_gap=og,fixed_gap=fg,ruler=ruler,
                    control_kind='oracle_block' if card.get('block_family')=='oracle' else 'ground_truth_cache_sentinel',
                    passed=negative_control(og,fg,ruler)))
        assert any(c['control_kind']=='oracle_block' for c in controls)
        groups={role:dict(n=sum(x['role']==role for x in rows),
            n_consequence=sum(x['role']==role and x['consequence'] is True for x in rows),
            n_undefined=sum(x['role']==role and x['consequence'] is None for x in rows))
            for role in ['random_block','real','trivial','other_block']}
        result=dict(h_s=h,pi_hash=ref.pi_hash,n_dev_eligible=len(dev),n_test_eligible=len(cohort),
            dev_tercile_cuts_s=cuts.tolist(),MRD_plaus=mrd,ruler=ruler,groups=groups,rows=rows,
            oracle_negative_controls=controls,**checked_gate(groups,controls))
        by.append(result)
        print(f'[G3 v5 VLM={ctx.include_vlm}] H={h} {groups} controls={controls} status={result["status"]}',flush=True)
    primary=next(x for x in by if x['h_s']==10.)
    return dict(by_h=by,primary_h=10.,D2_visible_end_cap_s=CAP,pass_gate=primary['pass_gate'],
        computation_valid=primary['computation_valid'],status=primary['status'],
        oracle_control_policy='v5 oracle block plus the card-declared ground-truth cache sentinel; both reported separately',
        missing_policy='BOT/wrong; no imputation; visible end=min(L+,22.0); fixed at H',
        **ctx.provenance())


def g3(root, include_vlm=True):
    root=Path(root)
    primary=Context(root,False)
    old=r2.read_json(primary.out/'gates/g3.json')
    without=dict(original_rule=old['original_rule'],v4_23p5=old['revised_rule'],v5_22p0=corrected_g3(primary))
    result=dict(primary=without,secondary=None,record_rule='v5_22p0',primary_pool='without_vlm',
                primary_h=10.,source_primary_history='outputs/r2b/gates/g3.json')
    if include_vlm:
        secondary=Context(root,True)
        r2b.g3(secondary)
        history=r2.read_json(secondary.out/'gates/g3.json')
        result['secondary']=dict(original_rule=history['original_rule'],v4_23p5=history['revised_rule'],v5_22p0=corrected_g3(secondary))
    result['pass_gate']=without['v5_22p0']['pass_gate']
    result['status']=without['v5_22p0']['status']
    r2.dump(result,root/'outputs/r2c/gates/g3_v5.json')


def run(root,task='all'):
    if task=='prepare': return prepare(root)
    if task=='g3': return g3(root)
    if task=='g3_primary': return g3(root,False)
    ctx=Context(root)
    tasks={'eval':evaluate,'calibrate':calibrate,'g2':r2b.g2,'g4':r2.g4,
        'a1':r2.a1,'a3':r2.a3,'a4':lambda c:r2.rank_ablation(c,'a4'),
        'a5':lambda c:r2.rank_ablation(c,'a5'),'mechanisms':r2.mechanisms,'g5':r2.g5}
    chosen=tasks.items() if task=='all' else [(task,tasks[task])]
    for name,fn in chosen:
        print('[stage] '+name,flush=True);fn(ctx)
    if task=='all':g3(root)
