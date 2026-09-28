"""Select every R2 arm using train/dev only, then seal all choices before test."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import datetime,hashlib,json,time
import joblib,numpy as np,pandas as pd,torch
from systems import common as C
from systems import postproc,commit
from systems.r2_models import STATE,JS,write_json,check_freeze,infer
from systems.r2_stops import endpoint_summary,ringel_calibrate,select_teaser,apply_ringel,apply_teaser


def child_id(base,rule):
    stem,seed=base.rsplit('__seed',1)
    return f'{stem}__{rule}__seed{seed}'


def spec(ck,rule,family,params=None,value=None):
    base=ck['system_id']
    sid=base if family in ['clip','prefix'] else child_id(base,rule)
    return {'system_id':sid,'parent_system_id':None if sid==base else base,'family':family,
            'arm_rule':{'mean':'mean_linear','msp':'commit_msp','margin':'commit_margin','ringel':'ringel2024_conditional_adapt','teaser':'teaser_ocsvm_adapt'}.get(rule,rule),
            'arm_value':value,'backbone':ck['backbone'],'seed':ck['seed'],'model_kind':ck['kind'],
            'dev_tuned_params':params or {},'checkpoint':str(Path(C.CKPT_DIR)/(base+'.pt'))}


def main():
    torch.set_num_threads(4);freeze=check_freeze();started=time.time()
    sealed=STATE/'selection_sealed.json'
    if sealed.exists(): print('SELECTION_ALREADY_SEALED');return
    training=json.loads((STATE/'training_complete.json').read_text());assert training['completed']==18
    man=C.load_manifest();subset=man[man.split.isin(['train','dev'])].sort_values('video_id').reset_index(drop=True)
    assert len(subset)==513
    train=subset.split.eq('train').to_numpy();dev=subset.split.eq('dev').to_numpy()
    y=subset.class_code.to_numpy();pl=subset.post_anchor_length_s.to_numpy()
    dv_y=y[dev];dv_pl=pl[dev];dv_elig=dv_pl+1e-9>=10
    dev_clusters=subset.loc[dev,'source_cluster_id'].to_numpy()
    clean=subset[['video_id','path','anchor_s']].copy()
    systems=[];hashes={};causality=[]
    cache=STATE/'development_predictions';cache.mkdir(exist_ok=True)
    for ckpath in training['selected_checkpoints']:
        ck=torch.load(ckpath,map_location='cpu',weights_only=False);base=ck['system_id']
        hashes[ckpath]=hashlib.sha256(Path(ckpath).read_bytes()).hexdigest()
        npz=cache/(base+'.npz')
        if npz.exists():
            with np.load(npz) as z: vids,pred,probs=z['video_ids'],z['preds'],z['probs']
        else:
            vids,pred,probs=infer(ck,clean)
            np.savez(npz,video_ids=vids,preds=pred,probs=probs)
        assert vids.tolist()==subset.video_id.tolist()
        base_dev=endpoint_summary(pred[dev],dv_y,dv_pl,JS)
        assert abs(base_dev['end_macro_acc']-ck['train_info']['best_dev_end_macro_acc'])<1e-9,(base,base_dev,ck['train_info']['best_dev_end_macro_acc'])
        base_spec=spec(ck,ck['kind'],'clip' if ck['kind']=='mean' else 'prefix',{'best_epoch':ck['train_info']['best_epoch'],**ck['hparams']})
        base_spec['dev_summary']=base_dev;systems.append(base_spec)
        # Recompute aligned prefixes at coarse steps on dev, never on test.
        dv_clean=clean.loc[dev].reset_index(drop=True)
        for stride in [2,4]:
            keep=JS%stride==0
            ids,re_pred,re_probs=infer(ck,dv_clean,JS[keep]//stride,.25*stride)
            mismatches=int((re_pred!=pred[dev][:,keep]).sum())
            err=float(np.nanmax(np.abs(re_probs-probs[dev][:,keep])))
            assert mismatches==0 and err<1e-6
            causality.append({'system_id':base,'stride':stride,'dev_clips':len(ids),'cells':int(re_pred.size),'prediction_mismatches':mismatches,'max_probability_error':err})
        if ck['kind']=='gru512':
            for rule,grid in postproc.GRIDS.items():
                best=None;scan=[]
                for value in grid:
                    pp,_=postproc.apply_arm(rule,probs[dev],value)
                    summary=endpoint_summary(pp,dv_y,dv_pl,JS)
                    feasible=summary['end_macro_acc']>=base_dev['end_macro_acc']-1e-9
                    scan.append({'value':value,'feasible':bool(feasible),**summary})
                    key=(0 if feasible else 1,summary['rmscd'])
                    if best is None or key<best[0]: best=(key,value,summary,feasible)
                _,value,summary,feasible=best
                item=spec(ck,rule,'postproc',{'value':value,'grid':grid,'constraint_satisfied':bool(feasible),'selected_at_delta_s':.25,'H_used_for_selection':10,'dev_scan':scan},value)
                item['dev_summary']=summary;systems.append(item)
            for rule,grid in [('msp',commit.MSP_LEVELS),('margin',commit.MARGIN_LEVELS)]:
                for value in grid:
                    p,cm,first=commit.apply_commit(probs[dev],rule,value,int(np.searchsorted(JS,0)))
                    item=spec(ck,rule+format(value,'g').replace('.','p'),'commit',{'rule':rule,'threshold':value,'threshold_sweep':True,'grid':grid,'selected_at_delta_s':.25},value)
                    item['arm_rule']='commit_'+rule;item['dev_summary']=endpoint_summary(p,dv_y,dv_pl,JS);systems.append(item)
        if ck['kind']!='mean':
            dprobs=probs[dev];win=(JS>=0)&(JS<=40)
            thresholds,rinfo=ringel_calibrate(dprobs[dv_elig][:,win],dv_y[dv_elig],dev_clusters[dv_elig],ck['seed'])
            rpred,_cm,_first=apply_ringel(dprobs,JS,thresholds)
            ritem=spec(ck,'ringel','commit',rinfo)
            ritem['dev_summary']=endpoint_summary(rpred,dv_y,dv_pl,JS)
            bank,tinfo=select_teaser(probs[train],y[train],pl[train],dprobs,dv_y,dv_pl,JS)
            tpred,_cm,_first=apply_teaser(dprobs,JS,bank,tinfo['v'])
            titem=spec(ck,'teaser','commit',tinfo);titem['dev_summary']=endpoint_summary(tpred,dv_y,dv_pl,JS)
            models=STATE/'stopping_models';models.mkdir(exist_ok=True)
            modelpath=models/(base+'.joblib')
            joblib.dump({'ringel_thresholds':thresholds,'teaser_bank':bank,'teaser_v':tinfo['v']},modelpath)
            for item in [ritem,titem]: item['stopping_model_path']=str(modelpath);systems.append(item)
            hashes[str(modelpath)]=hashlib.sha256(modelpath.read_bytes()).hexdigest()
        write_json(STATE/'selection_progress.json',{'last_parent':base,'fine_system_count':len(systems)})
        print('SELECTED_PARENT',base,'fine_system_count',len(systems),'base_dev',base_dev['end_macro_acc'],flush=True)
    assert len(systems)==120 and len({x['system_id'] for x in systems})==120
    write_json(STATE/'subsample_checks.json',causality)
    for f in Path(__file__).parent.glob('r2_*.py'): hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    plan={'freeze_commit':freeze['commit'],'sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fine_systems':systems,'fine_count':120,'coarse_count':204,'hashes':hashes,'test_metrics_computed':0,'wall_seconds':time.time()-started}
    write_json(sealed,plan)
    print('ALL_SELECTION_SEALED',len(systems),'test_metrics',0,flush=True)


if __name__=='__main__': main()
