import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,numpy as np,torch
from systems import common as C
from systems.r2_models import STATE,infer,JS,write_json
from systems.r2_stops import endpoint_summary
torch.set_num_threads(4)
man=C.load_manifest();dv=man[man.split.eq('dev')].sort_values('video_id')
items=json.loads((STATE/'training_progress.json').read_text())['selected_checkpoints'];checks=[]
for name in items:
    ck=torch.load(name,map_location='cpu',weights_only=False)
    vids,p,q=infer(ck,dv[['video_id','path','anchor_s']])
    score=endpoint_summary(p,dv.class_code.to_numpy(),dv.post_anchor_length_s.to_numpy(),JS)['end_macro_acc']
    saved=ck['train_info']['best_dev_end_macro_acc']
    print(ck['system_id'],'dev inference',score,'saved selection score',saved,flush=True)
    assert abs(score-saved)<1e-9
    checks.append({'system_id':ck['system_id'],'score_consistent':True})
write_json(STATE/'dev_endpoint_consistency.json',checks)
