"""Generate the sealed R2 library; test labels are not passed to any model/rule."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import hashlib,json,time,datetime
import joblib,numpy as np,pandas as pd,torch
from systems import common as C
from systems import postproc,commit
from systems.r2_models import STATE,JS,infer,check_freeze,write_json
from systems.r2_stops import apply_ringel,apply_teaser


def assert_seal(plan):
    check_freeze()
    for name,digest in plan['hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'Sealed input changed: '+name


def make_card(item,stride,plan):
    card=dict(item);card.pop('checkpoint',None)
    card['system_id']=item['system_id']+('' if stride==1 else f'__stride{stride}')
    card.update(causal=True,trained_on_split='train',train_data_unknown=item['backbone']=='clipb16',
                description='R2 preregistered balanced classifier or causal rule; rule and backbone define identity',
                delta_s=.25*stride,stride_from_finest=stride,
                cost_note='Fixed pretrained 4 fps encoder; cached causal prefixes; weights trained only on train; all selection on dev',
                subsampling_equivalent=item['family'] in ['clip','prefix'],
                pre_anchor_rows='Raw base predictions; commitment may not fire at negative j; protocol layer masks negative delta',
                freeze_commit=plan['freeze_commit'],selection_sealed_utc=plan['sealed_utc'],
                subsampling_note='Causal prefix function' if item['family'] in ['clip','prefix'] else 'Regenerated at this delta from the base trajectory, using the fine-grid selected rule and parameters')
    card['dev_tuned_params']={**card['dev_tuned_params'],'selected_at_delta_s':.25,'H_used_for_selection':10.0}
    return card


def main():
    torch.set_num_threads(4);started=time.time()
    plan=json.loads((STATE/'selection_sealed.json').read_text());assert_seal(plan)
    man=C.load_manifest().sort_values('video_id').reset_index(drop=True)
    # A clean view supplies neither label, duration nor remaining time to inference.
    test_view=man.loc[man.split.eq('test'),['video_id','path','anchor_s']].copy()
    assert len(test_view)==1514
    devtrain_ids=man.loc[~man.split.eq('test'),'video_id'].tolist()
    id_order=man.video_id.tolist();registry=[]
    bases=[it for it in plan['fine_systems'] if it['parent_system_id'] is None]
    testcache=STATE/'test_predictions';testcache.mkdir(exist_ok=True)
    for baseitem in bases:
        base=baseitem['system_id'];ck=torch.load(baseitem['checkpoint'],map_location='cpu',weights_only=False)
        cache=testcache/(base+'.npz');marker=testcache/(base+'.started.json')
        if cache.exists():
            with np.load(cache) as z: tvids,tpred,tprobs=z['video_ids'],z['preds'],z['probs']
        else:
            if marker.exists(): raise RuntimeError('Interrupted test inference; no automatic second pass: '+base)
            with marker.open('x') as f: json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_inference_pass':1,'test_metrics_computed':0},f)
            tvids,tpred,tprobs=infer(ck,test_view)
            tmp=cache.with_suffix('.npz.partial')
            with tmp.open('wb') as f: np.savez(f,video_ids=tvids,preds=tpred,probs=tprobs)
            os.replace(tmp,cache)
        with np.load(STATE/'development_predictions'/(base+'.npz')) as z:
            dvids,dpred,dprobs=z['video_ids'],z['preds'],z['probs']
        joined=list(dvids)+list(tvids);idx={v:i for i,v in enumerate(joined)};order=[idx[v] for v in id_order]
        pred=np.concatenate([dpred,tpred])[order];probs=np.concatenate([dprobs,tprobs])[order]
        children=[it for it in plan['fine_systems'] if it['parent_system_id']==base]
        stopping=None
        if ck['kind']!='mean': stopping=joblib.load(STATE/'stopping_models'/(base+'.joblib'))
        for item in [baseitem]+children:
            strides=[1] if item['family'] in ['clip','prefix'] else [1,2,4]
            for stride in strides:
                keep=JS%stride==0;j=JS[keep]//stride;p=probs[:,keep];raw=pred[:,keep]
                rule=item['arm_rule'];params=item['dev_tuned_params'];cm=None
                if item['family'] in ['clip','prefix']: out,q=raw,p
                elif item['family']=='postproc': out,q=postproc.apply_arm(rule,p,params['value'])
                elif rule in ['commit_msp','commit_margin']:
                    out,cm,_=commit.apply_commit(p,params['rule'],params['threshold'],int(np.searchsorted(j,0)));q=p
                elif rule=='ringel2024_conditional_adapt':
                    out,cm,_=apply_ringel(p,j,stopping['ringel_thresholds'],.25*stride);q=p
                elif rule=='teaser_ocsvm_adapt':
                    out,cm,_=apply_teaser(p,j,stopping['teaser_bank'],stopping['teaser_v'],.25*stride);q=p
                else: raise ValueError(rule)
                card=make_card(item,stride,plan);sid=card['system_id']
                target=Path(C.ANSWERS_DIR)/sid;target.mkdir(exist_ok=True)
                if not (target/'complete.json').exists():
                    frame=C.build_answer_frame(id_order,out,q,delta_s=.25*stride,committed=cm,j_offset=int(j[0]))
                    partial=target/'answers.csv.partial';frame.to_csv(partial,index=False);os.replace(partial,target/'answers.csv')
                    C.write_card(str(target),card)
                    if 'dev_scan' in params: pd.DataFrame(params['dev_scan']).to_csv(target/'dev_scan.csv',index=False)
                    write_json(target/'complete.json',{'selection_sealed_utc':plan['sealed_utc'],'rows':len(frame),'test_metrics_computed':0})
                registry.append({'system_id':sid,'arm_rule':rule,'arm_value':item['arm_value'],'family':item['family'],'backbone':item['backbone'],'seed':item['seed'],'model_kind':item['model_kind'],'stride':stride,'delta_s':.25*stride,'parent_system_id':item['parent_system_id'],'answers_dir':str(target.resolve())})
        print('GENERATED_PARENT',base,'directories',len(registry),'test_metrics',0,flush=True)
    assert len(registry)==324 and len({x['system_id'] for x in registry})==324
    write_json(STATE/'new_registry.json',registry)
    pd.DataFrame(registry).to_csv(STATE/'new_registry.csv',index=False)
    write_json(STATE/'generation_complete.json',{'n_directories':len(registry),'wall_seconds':time.time()-started,'test_metrics_computed':0,'test_inference_passes_per_base':1})
    print('ALL_GENERATED',len(registry),flush=True)


if __name__=='__main__': main()
