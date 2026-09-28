"""Actual dev-prefix truncation check; no test clips or test metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import json,numpy as np,torch
from systems import common as C
from systems.r2_models import STATE,JS,read_items,infer,write_json
from systems.train_prefix import CausalGRUClassifier
from systems.infer_answers import softmax_np


def main():
    torch.set_num_threads(4)
    man=C.load_manifest();dev=man[man.split.eq('dev')].sort_values('video_id').head(5)[['video_id','path','anchor_s']]
    training=json.loads((STATE/'training_progress.json').read_text());records=[]
    for path in training['selected_checkpoints']:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        ids,preds,probs=infer(ck,dev);items=read_items(dev,ck['features_dir'])
        if ck['kind']!='mean':
            model=CausalGRUClassifier(ck['in_dim'],int(ck['kind'][3:])).cuda();model.load_state_dict(ck['state_dict']);model.eval()
        errors=[];disagreements=0
        for i,it in enumerate(items):
            for fine_j in [0,4,20,40,80]:
                end=it['anchor']+fine_j*.25;sel=it['t']<=end+1e-9
                f=it['f'][sel].astype(np.float64)
                if not len(f): continue
                if ck['kind']=='mean':
                    z=((f.mean(0)-ck['feat_mean'])/ck['feat_std'])@ck['state_dict']['weight'].numpy().T+ck['state_dict']['bias'].numpy()
                    z=z[0]
                else:
                    x=torch.as_tensor((f.astype(np.float32)-ck['feat_mean'])/ck['feat_std'],device='cuda',dtype=torch.float32)[None]
                    with torch.inference_mode(): z,_=model(x)
                    z=z[0,-1].cpu().numpy()
                p=softmax_np(z);col=int(np.where(JS==fine_j)[0][0])
                errors.append(float(np.max(np.abs(p-probs[i,col]))));disagreements+=int(p.argmax()!=preds[i,col])
        assert disagreements==0 and max(errors)<1e-5,(ck['system_id'],max(errors))
        records.append({'system_id':ck['system_id'],'prefixes_compared':len(errors),'pred_disagreements':disagreements,'max_probability_error':max(errors)})
        print('DEV_PREFIX_TRUNCATION_OK',records[-1],flush=True)
    write_json(STATE/'real_prefix_checks.json',records)


if __name__=='__main__': main()
