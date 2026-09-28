"""K-b full library integration, preregistered v4 gates and audit outputs."""
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
import yaml
from . import r2
from .r2b_identity import metadata, group_key
from .cli import _intervals_from_samples, _phenomena_block, _track_of, cmd_report_json
from .metrics import alt_cohort_report, stable_correct, first_stable_index
from .protocol import TOL
from .cohort import cluster_codes

class Context(r2.Context):
    def __init__(self,root):
        super().__init__(root,Path(root)/'outputs/r2b',(184,172,168))
        self.meta={s:metadata(s,self.cards) for s in self.ids}
    def provenance(self):
        return dict(super().provenance(),methods='REPORT_R2_Kb.md',freeze_version='v4',vlm_included=False)

def prepare(root):
    root=Path(root);out=root/'outputs';dest=out/'r2b';dest.mkdir(exist_ok=True)
    old_index=out/'r2_Ka/report_index.json'
    if not old_index.exists(): old_index=out/'report_index.json'
    old=r2.read_json(old_index)
    old_ids=sorted({x['system_id'] for x in old['rows'] if not x['system_id'].startswith('r2__')})
    assert len(old_ids)==64
    registry=pd.read_csv(out/'r2_S/new_registry.csv')
    new=registry[registry.stride==1].system_id.tolist()
    assert len(new)==120 and len(registry)==324
    ids=sorted(old_ids+new)
    cards={s:yaml.safe_load((out/'answers'/s/'system_card.yaml').read_text()) for s in ids}
    rows=[];stands=[]
    for s in ids:
        card=cards[s];assert card['system_id']==s
        meta=metadata(s,cards)
        for d in [.5,1.]:
            if card.get('subsampling_equivalent',True) is False:
                p=out/'answers'/f'{s}__stride{int(d/.25)}'
                assert p.is_dir(),p
                sc=yaml.safe_load((p/'system_card.yaml').read_text())
                assert float(sc.get('delta_s',sc.get('grid',{}).get('delta_s')))==d
                stands.append(str(p.resolve()))
        for h,ph in [(4.,'77200b351bf1'),(10.,'8ac32aae418b'),(21.5,'3fb251dc210c')]:
            rows.append(dict(system_id=s,h_s=h,pi_hash=ph,**meta))
    assert len(stands)==284
    r2.dump(dict(n_rows=len(rows),rows=rows),dest/'report_index.json')
    r2.dump(dict(system_ids=ids,n_systems=len(ids),n_old_nonblock=52,n_r2=120,n_blocks=12,
                 n_stride_stand_ins=len(stands),stand_ins=stands,identity_source='system_card.yaml',
                 grouping_key=['library_round','backbone','model_kind','arm_rule','commit_threshold']),dest/'library.json')
    print('[prepare] 184 systems, 284 stand-ins',flush=True)

def evaluate(ctx):
    for h in ctx.cfg.h_list_s:
        ref=ctx.cfg.pi0(h);first=ctx.eval(ctx.ids[0],ref)
        reps,w=r2.bootstrap_weights(cluster_codes(pd.DataFrame({'source_cluster_id':first.source_cluster_id})),ctx.cfg.bootstrap_n,ctx.cfg.seed)
        for s in ctx.ids:
            res=ctx.eval(s,ref)
            assert np.array_equal(res.video_ids,first.video_ids)
            points=res.frozen_family(ctx.cfg.s_report_delta_s)
            samples=r2.family_bootstrap(res,reps,w)
            payload=res.to_json_dict(ctx.cfg.s_report_delta_s)
            payload.update(ctx.meta[s]);payload.update(audit_split='test',n_clusters=len(np.unique(first.source_cluster_id)),
                track=_track_of(ctx.test),family=ctx.cards[s]['family'],group_key=group_key(ctx.meta[s]))
            payload['bootstrap']=_intervals_from_samples(points,samples,ctx.cfg.alpha_pairs,len(reps))
            payload['commit']['tau_c']=payload['commit']['tau_c_mean']
            table=ctx.table(s,ref.delta_s)
            payload['alt_cohorts']=alt_cohort_report(ctx.test,table,ref,ctx.cfg.grid_max_s,ctx.cfg.s_report_delta_s[0])
            payload['phenomena']=_phenomena_block(res,ctx.test,table,ctx.cfg)
            r2.dump(payload,ctx.out/'metrics'/ref.pi_hash/f'{s}.json')
        print(f'[eval] H={h}, systems={len(ctx.ids)}, N={len(first.video_ids)}',flush=True)
    from types import SimpleNamespace
    cmd_report_json(SimpleNamespace(metrics=str(ctx.out/'metrics'),out=str(ctx.out/'report_index.json')))
    # The main index points explicitly to K-b metrics; old metric trees remain intact.
    index=r2.read_json(ctx.out/'report_index.json');index['metrics_root']='r2b/metrics';index['outputs_root']='r2b'
    for row in index['rows']: row.update(ctx.meta[row['system_id']])
    r2.dump(index,ctx.out/'report_index.json')
    r2.dump(index,ctx.root/'outputs/report_index.json')
    pd.DataFrame(index['rows']).to_csv(ctx.out/'report_index.csv',index=False)
    pd.DataFrame(index['rows']).to_csv(ctx.root/'outputs/report_index.csv',index=False)

def g2_metric(metric,min_r,mrd,ruler,r0=.9):
    degenerate=ruler is None or not np.isfinite(ruler) or ruler<=0
    excluded=metric=='median_flips'  # frozen F(Delta) trivial dependence never carries G2
    return dict(metric=metric,min_R=min_r,MRD_plaus=mrd,ruler=ruler,
        R_arm=bool(min_r<r0) if not excluded else None,
        MRD_arm=bool(mrd>=ruler) if not excluded and not degenerate else None,
        ruler_degenerate=degenerate,excluded_trivial_dependence=excluded)

def g2(ctx):
    by=[]
    for h in ctx.cfg.h_list_s:
        cal=r2.read_json(ctx.out/'calib_plaus'/ctx.cfg.pi0(h).pi_hash/'calibration.json')
        rows=[g2_metric(m,cal['metrics'][m]['min_R'],cal['metrics'][m]['MRD_plaus'],cal['rulers_by_metric'][m],ctx.cfg.r0) for m in r2.METRICS]
        by.append(dict(h_s=h,rows=rows,pass_gate=any(x['R_arm'] is True or x['MRD_arm'] is True for x in rows)))
    ctx.save('g2',dict(by_h=by,primary_h=10.,pass_gate=next(x['pass_gate'] for x in by if x['h_s']==10.),
        rule='v4 A4-1; RMSCD MRD restricted to equal H; F(Delta) excluded'))

def bootstrap_gap(values,lengths,cuts,w):
    out=r2.three_strata(values,lengths,cuts)
    labs=np.searchsorted(cuts,lengths,side='left')
    if out['gap_long_minus_short'] is None:
        return dict(out,ci_lo=None,ci_hi=None,n_boot_valid=0)
    means=[]
    for k in [0,2]:
        mask=(labs==k).astype(float);den=w@mask
        means.append(np.divide(w@(np.asarray(values)*mask),den,out=np.full(len(w),np.nan),where=den>0))
    diffs=means[1]-means[0];diffs=diffs[np.isfinite(diffs)]
    return dict(out,ci_lo=float(np.quantile(diffs,.025)),ci_hi=float(np.quantile(diffs,.975)),n_boot_valid=len(diffs))

def revised_consequence(own,fixed,mrd,ruler):
    vals=[own.get(k) for k in ['gap_long_minus_short','ci_lo','ci_hi']]+[fixed.get(k) for k in ['gap_long_minus_short','ci_lo','ci_hi']]+[mrd,ruler]
    if any(v is None or not np.isfinite(v) for v in vals): return None
    return bool((own['ci_lo']>0 or own['ci_hi']<0) and abs(own['gap_long_minus_short'])>mrd and
                (fixed['ci_lo']<=0<=fixed['ci_hi'] or abs(fixed['gap_long_minus_short'])<ruler))

def own_visible_delays(manifest,table,delta,cap=23.5):
    offsets=np.arange(int(np.floor(cap/delta))+1)*delta
    js=np.broadcast_to(np.rint(offsets/table.delta_s).astype(int),(len(manifest),len(offsets)))
    pred,present,_,_=table.lookup_2d(manifest.video_id,js)
    visible=np.minimum(manifest.post_anchor_length_s.to_numpy(),cap)
    inside=offsets[None,:]<=visible[:,None]+TOL
    correct=(pred==manifest.class_code.to_numpy()[:,None]) & present
    indicator=stable_correct(correct|~inside)
    valid=indicator & inside
    first=first_stable_index(valid)
    # Never stable: censor at each clip's visible end, never at reference H.
    delay=np.where(first>=0,np.minimum(first*delta,visible),visible)
    return delay,dict(n_missing_visible=int(np.sum(inside & ~present)),n_visible_cells=int(inside.sum()),
                      n_clips_with_missing_visible=int(np.any(inside & ~present,axis=1).sum()))

def g3(ctx):
    r2.g3(ctx)  # Preserve the exact K-a implementation of original R2-8.
    original=r2.read_json(ctx.out/'gates/g3.json');revised=[]
    for h in ctx.cfg.h_list_s:
        ref=ctx.cfg.pi0(h);cohort=ctx.test[ctx.test.post_anchor_length_s>=h-TOL].reset_index(drop=True)
        dev=ctx.dev[ctx.dev.post_anchor_length_s>=h-TOL]
        cuts=np.quantile(dev.post_anchor_length_s,[1/3,2/3])
        _,w=r2.bootstrap_weights(cluster_codes(cohort),ctx.cfg.bootstrap_n,ctx.cfg.seed)
        cal=r2.read_json(ctx.out/'calib_plaus'/ref.pi_hash/'calibration.json')
        mrd=cal['metrics']['RMSCD@H']['MRD_plaus'];ruler=cal['rulers_by_metric']['RMSCD@H'];rows=[]
        for s in ctx.ids:
            res=ctx.eval(s,ref);assert np.array_equal(res.video_ids,cohort.video_id)
            ownv,coverage=own_visible_delays(cohort,ctx.table(s,ref.delta_s),ref.delta_s)
            own=bootstrap_gap(ownv,cohort.post_anchor_length_s.to_numpy(),cuts,w)
            fixed=bootstrap_gap(r2.rmscd_per_video(res.indicator,res.delta_s),cohort.post_anchor_length_s.to_numpy(),cuts,w)
            role='random_block' if ctx.cards[s].get('block_family')=='rand' else 'other_block' if s in ctx.blocks else 'real' if s in ctx.nontrivial else 'trivial'
            rows.append(dict(system_id=s,role=role,own=own,fixed=fixed,MRD_plaus=mrd,ruler=ruler,
                consequence=revised_consequence(own,fixed,mrd,ruler),coverage=coverage))
        groups={role:dict(n=sum(x['role']==role for x in rows),n_consequence=sum(x['role']==role and x['consequence'] is True for x in rows),n_undefined=sum(x['role']==role and x['consequence'] is None for x in rows)) for role in ['random_block','real','trivial','other_block']}
        passed=(groups['random_block']['n_consequence']==2 and groups['real']['n_consequence']>=.5*groups['real']['n'])
        revised.append(dict(h_s=h,n_dev_eligible=len(dev),n_test_eligible=len(cohort),dev_tercile_cuts_s=cuts.tolist(),MRD_plaus=mrd,ruler=ruler,groups=groups,rows=rows,pass_gate=passed))
        print(f'[G3 revised] H={h}: {groups}',flush=True)
    original['original_rule']={k:original[k] for k in ['by_h','pass_gate','dev_tercile_cuts_s','D2_visible_end_cap_s']}
    original['revised_rule']=dict(by_h=revised,primary_h=10.,pass_gate=next(x['pass_gate'] for x in revised if x['h_s']==10.),D2_visible_end_cap_s=23.5,
        missing_policy='Missing visible answer cells remain BOT/wrong; report their coverage; no regenerated or imputed answers.',
        censoring='Own stable delay censored at min(L+,23.5), fixed at H; same E_H and shared cluster replicates.')
    original.update(pass_gate_original=original['pass_gate'],pass_gate_revised=original['revised_rule']['pass_gate'],rule_reporting='original_and_v4_revised')
    r2.dump(original,ctx.out/'gates/g3.json')

def run(root,task='all'):
    if task=='prepare': return prepare(root)
    ctx=Context(root)
    tasks={'eval':evaluate,'calibrate':r2.calibrate,'g2':g2,'g3':g3,'g4':r2.g4,'a1':r2.a1,'a3':r2.a3,
           'a4':lambda c:r2.rank_ablation(c,'a4'),'a5':lambda c:r2.rank_ablation(c,'a5'),'mechanisms':r2.mechanisms,'g5':r2.g5}
    selected=tasks.items() if task=='all' else [(k,v) for k,v in tasks.items() if k!='eval'] if task=='remaining' else [(k,v) for k,v in tasks.items() if k in ['a1','a3','a4','a5','mechanisms','g5']] if task=='controls' else [(task,tasks[task])]
    for name,fn in selected:
        print('[stage] '+name,flush=True);fn(ctx)
