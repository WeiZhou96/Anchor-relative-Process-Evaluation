"""Full structural acceptance of every new answer directory without label metrics."""
from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd,yaml
from systems import common as C
from systems import verify_answers as V
from systems.r2_models import STATE,write_json,check_freeze
from systems.r2_generate import assert_seal


def main():
    started=time.time();freeze=check_freeze()
    plan=json.loads((STATE/'selection_sealed.json').read_text());assert_seal(plan)
    registry=json.loads((STATE/'new_registry.json').read_text());assert len(registry)==324
    man=C.load_manifest()[['video_id']];expected=set(man.video_id)
    summaries=[];total_rows=0
    for item in registry:
        sid=item['system_id'];target=Path(item['answers_dir']);path=target/'answers.csv'
        V.verify(sid,str(path),man)
        df=pd.read_csv(path,usecols=['video_id','j','pred','committed']).sort_values(['video_id','j'])
        assert set(df.video_id)==expected
        assert not df.duplicated(['video_id','j']).any()
        js=np.sort(df.j.unique());pred=df.pred.to_numpy().reshape(2027,len(js));com=df.committed.to_numpy().reshape(2027,len(js))
        assert len(df)==2027*(100//item['stride'])
        if item['family']=='commit':
            assert not com[:,js<0].any()
            assert not np.any(com[:,:-1]&~com[:,1:])
            for i in range(2027):
                on=np.where(com[i])[0]
                if len(on): assert np.all(pred[i,on]==pred[i,on[0]]) and pred[i,on[0]]>=0
            assert np.all(pred[(js>=0)[None,:]&~com]==-1)
        elif item['family'] in ['clip','prefix']: assert np.all(pred[:,js>=0]>=0)
        card=yaml.safe_load((target/'system_card.yaml').read_text())
        assert card['system_id']==sid and card['freeze_commit']==freeze['commit']
        assert card['arm_rule']==item['arm_rule'] and card['backbone']==item['backbone']
        assert card['seed']==item['seed'] and card['dev_tuned_params']['selected_at_delta_s']==.25
        summaries.append({'system_id':sid,'rows':len(df),'n_clips':2027,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        total_rows+=len(df)
    assert not V.FAILS,V.FAILS
    old=[]
    for path in sorted(Path(C.ANSWERS_DIR).glob('*/answers.csv')):
        if path.parent.name.startswith('r2__'): continue
        card=yaml.safe_load((path.parent/'system_card.yaml').read_text()) or {}
        old.append({'system_id':path.parent.name,'answers_dir':str(path.parent.resolve()),'family':card.get('family'),'stride':card.get('stride_from_finest',1)})
    write_json(STATE/'all_answers_paths.json',{'new':registry,'old':old})
    (STATE/'all_answers_paths.txt').write_text('\n'.join(x['answers_dir'] for x in old+registry)+'\n')
    write_json(STATE/'verification_complete.json',{'new_directories':len(registry),'old_directories_indexed':len(old),'new_rows':total_rows,'checks_failed':len(V.FAILS),'all_new_clip_coverage':2027,'wall_seconds':time.time()-started,'test_metrics_computed':0,'files':summaries})
    print('R2_VERIFY_OK',len(registry),'new_directories','rows',total_rows,'old_paths',len(old),flush=True)


if __name__=='__main__': main()
