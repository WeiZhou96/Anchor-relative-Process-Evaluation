from pathlib import Path
import json,os,time,datetime
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from systems import common as C
from systems.r2_models import STATE,SEEDS,train_one,check_freeze,write_json


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    freeze=check_freeze();started=time.time()
    if not (STATE/'training_start.json').exists():
        write_json(STATE/'training_start.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_commit':freeze['commit'],'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'test_metrics_computed':0})
    selected=[]
    for backbone,features_dir in [('r18',C.FEATURES_DIR),('second',None)]:
        if backbone=='second':
            while True:
                statepath=STATE/'feature_backend.json'
                if statepath.exists():
                    state=json.loads(statepath.read_text())
                    if state.get('complete'): break
                print('WAITING_FOR_SECOND_BACKBONE_FEATURES',flush=True)
                time.sleep(30)
            backbone=state['backend'];features_dir=state['features_dir']
        for kind in ['mean','gru128','gru512']:
            for seed in SEEDS:
                path=train_one(backbone,kind,seed,features_dir);selected.append(str(path))
                write_json(STATE/'training_progress.json',{'selected_checkpoints':selected,'completed':len(selected),'expected':18})
    write_json(STATE/'training_complete.json',{'selected_checkpoints':selected,'completed':18,'wall_seconds':time.time()-started,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_metrics_computed':0})
    print('ALL_TRAINING_DONE',len(selected),flush=True)


if __name__=='__main__': main()
