"""Frozen-model input-cadence contrast with a common output-query grid."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'anchor_shift_20260921'))
from run_anchor_shift import ARMS, NewFrames, PrefixModel, dump, sha

CONDITIONS = {'stride2':list(range(0,65,2)), 'stride4':list(range(0,65,4)),
              'stride8':list(range(0,65,8)), 'legacy4plus39':sorted(set(range(0,65,4))|{39})}
QUERIES = list(range(0,65,8))


def infer(model,data,mode,offsets):
    selected=[offsets.index(q) for q in QUERIES]
    outputs=[]
    with torch.inference_mode():
        for batch in data.split(128):
            padded=torch.zeros((128,max(19,len(offsets)),512),device='cuda')
            padded[:len(batch),:len(offsets)]=batch
            if mode=='anchor':
                logits=model(padded[:,:1]).expand(-1,len(QUERIES),-1)
            else:
                logits=model(padded)[:,selected]
            outputs.append(logits[:len(batch)].softmax(-1).cpu())
    return torch.cat(outputs).numpy()


def main(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    started=time.time()
    args.out.mkdir(parents=True,exist_ok=False)
    parent=args.project/'deliverables/pixel_pilot_20260920/model_run'
    controls=args.project/'deliverables/control_pilot_20260920/model_run'
    original=[json.loads(line) for line in (parent/'cohort.jsonl').read_text().splitlines()]
    indices=[i for i,r in enumerate(original) if r['split']=='dev']
    rows=[original[i] for i in indices]
    assert len(rows)==269 and all(r['anchor_frame']+64<=r['last_frame'] for r in rows)
    old_plan=json.loads((parent/'pilot_plan.json').read_text())
    old_offsets=old_plan['offsets_frames']
    union=sorted({d for offsets in CONDITIONS.values() for d in offsets})
    missing=[d for d in union if d not in old_offsets]
    assert missing==list(range(2,63,4))
    checkpoints=[(parent if arm in ('prefix_mean','gru') else controls)/f'{arm}_{seed}.pt' for arm,_,_ in ARMS for seed in old_plan['seeds']]
    plan=dict(conditions=CONDITIONS,output_offsets=QUERIES,horizon=64,output_delta=8,common_n=269,
        arms=ARMS,seeds=old_plan['seeds'],union_offsets=union,new_offsets=missing,
        estimand='inference-time input cadence of fixed checkpoints; state-update count and compute also change; not an optimally retrained model comparison',
        bridge='legacy4plus39 retains the old exceptional input at39; regular stride4 removes it and is NOT silently treated as the old baseline',
        static_control='same first frame and checkpoint across all input conditions',
        no_checkpoint_reselection=True,source_identity_certified=False,formal=False,
        source_sha256=sha(Path(__file__)),parent_features_sha256=sha(parent/'features.npz'),
        normalizer_sha256=sha(parent/'normalizer.npz'),weights_sha256=sha(args.weights),
        imported_helper_sha256=sha(Path(__file__).resolve().parent.parent/'anchor_shift_20260921/run_anchor_shift.py'),
        checkpoint_sha256={p.name:sha(p) for p in checkpoints})
    dump(args.out/'plan.json',plan)
    (args.out/'cohort.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cached=np.load(parent/'features.npz')
    assert np.array_equal(cached['video_ids'][indices],[r['video_id'] for r in rows])
    old=cached['features'][indices]
    dataset=NewFrames(rows,args.root,missing)
    backbone=models.resnet18(weights=None)
    backbone.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True))
    backbone.fc=nn.Identity()
    backbone=backbone.cuda().eval()
    chunks,digests=[],[]
    with torch.inference_mode():
        for images,hashes in DataLoader(dataset,batch_size=128,num_workers=4,shuffle=False,pin_memory=True):
            chunks.append(backbone(images.cuda()).cpu().numpy())
            digests.extend(hashes)
    added=np.concatenate(chunks).reshape(len(rows),len(missing),512)
    features=np.stack([old[:,old_offsets.index(d)] if d in old_offsets else added[:,missing.index(d)] for d in union],axis=1)
    np.savez_compressed(args.out/'input_features.npz',features=features,offsets=union,video_ids=np.array([r['video_id'] for r in rows]))
    (args.out/'new_image_inputs.jsonl').write_text(''.join(json.dumps(dict(path=str(p.relative_to(args.root)),sha256=h))+'\n' for p,h in zip(dataset.items,digests)))
    del backbone
    norm=np.load(parent/'normalizer.npz')
    x=torch.from_numpy((features-norm['mean'])/norm['std']).cuda()
    runs=[]
    for arm,kind,mode in ARMS:
        source=parent if arm in ('prefix_mean','gru') else controls
        for seed in plan['seeds']:
            model=PrefixModel(kind).cuda().eval()
            model.load_state_dict(torch.load(source/f'{arm}_{seed}.pt',map_location='cpu',weights_only=True))
            anchor_reference=None
            for condition,offsets in CONDITIONS.items():
                data=x[:,[union.index(d) for d in offsets]]
                # Queries through16 must be invariant to changes in all later input samples.
                changed=data[:2].clone()
                changed[:,offsets.index(16)+1:]=100
                p_original=infer(model,data[:2],mode,offsets)
                p_changed=infer(model,changed,mode,offsets)
                np.testing.assert_array_equal(p_original[:,:3],p_changed[:,:3])
                probabilities=infer(model,data,mode,offsets)
                file=f'{arm}_{seed}_{condition}_predictions.npz'
                np.savez_compressed(args.out/file,probabilities=probabilities,offsets=QUERIES,input_offsets=offsets,video_ids=np.array([r['video_id'] for r in rows]))
                result=dict(system=f'{arm}_{seed}',arm=arm,seed=seed,condition=condition,input_frames=len(offsets),predictions_file=file)
                if mode=='anchor':
                    if anchor_reference is None: anchor_reference=probabilities
                    np.testing.assert_array_equal(anchor_reference,probabilities)
                if condition=='legacy4plus39':
                    reference=np.load(source/f'{arm}_{seed}_predictions.npz')['probabilities'][indices][:,[old_offsets.index(q) for q in QUERIES]]
                    np.testing.assert_array_equal(reference,probabilities)
                    result['legacy_replay_max_error']=float(np.max(np.abs(reference-probabilities)))
                runs.append(result)
            print(f'inferred {arm}_{seed}, four input conditions',flush=True)
    dump(args.out/'summary.json',dict(runs=runs,new_images=len(dataset),reused_feature_positions=len(rows)*(len(union)-len(missing)),
        causality_checks_passed=True,static_invariance_passed=True,legacy_replay_exact=True,elapsed_seconds=time.time()-started))
    print('INPUT_STRIDE_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
