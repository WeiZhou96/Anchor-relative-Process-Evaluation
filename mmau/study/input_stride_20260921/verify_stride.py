"""Replay all input-cadence outputs and verify all new image bytes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_stride import PrefixModel, infer, sha, dump


def main(project,directory,root):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    first=project/'deliverables/pixel_pilot_20260920/model_run'
    controls=project/'deliverables/control_pilot_20260920/model_run'
    plan=json.loads((directory/'plan.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    assert sha(Path(__file__).parent/'run_stride.py')==plan['source_sha256']
    assert sha(first/'features.npz')==plan['parent_features_sha256']
    assert sha(first/'normalizer.npz')==plan['normalizer_sha256']
    rows=[json.loads(line) for line in (directory/'cohort.jsonl').read_text().splitlines()]
    original=[json.loads(line) for line in (first/'cohort.jsonl').read_text().splitlines()]
    assert rows==[r for r in original if r['split']=='dev']
    cache=np.load(directory/'input_features.npz')
    union=cache['offsets'].tolist()
    assert union==plan['union_offsets']
    assert np.array_equal(cache['video_ids'],[r['video_id'] for r in rows])
    previous=np.load(first/'features.npz')['features'][[i for i,r in enumerate(original) if r['split']=='dev']]
    old_offsets=json.loads((first/'pilot_plan.json').read_text())['offsets_frames']
    features=cache['features']
    for offset in union:
        if offset in old_offsets:
            np.testing.assert_array_equal(features[:,union.index(offset)],previous[:,old_offsets.index(offset)])
    inputs=[json.loads(line) for line in (directory/'new_image_inputs.jsonl').read_text().splitlines()]
    assert len(inputs)==4304
    for i,row in enumerate(rows):
        for j,offset in enumerate(plan['new_offsets']):
            item=inputs[i*16+j]
            assert str(Path(item['path']).parent)==row['relative_image_directory']
            assert sha(root/item['path'])==item['sha256']
            frame=int(Path(item['path']).stem)
            assert frame==row['anchor_frame']+offset and row['first_frame']<=frame<=row['last_frame']
    norm=np.load(first/'normalizer.npz')
    x=torch.from_numpy((features-norm['mean'])/norm['std']).cuda()
    verified=[]
    for arm,kind,mode in plan['arms']:
        source=first if arm in ('prefix_mean','gru') else controls
        for seed in plan['seeds']:
            checkpoint=source/f'{arm}_{seed}.pt'
            assert sha(checkpoint)==plan['checkpoint_sha256'][checkpoint.name]
            model=PrefixModel(kind).cuda().eval()
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
            for condition,offsets in plan['conditions'].items():
                run=next(r for r in summary['runs'] if r['system']==f'{arm}_{seed}' and r['condition']==condition)
                path=directory/run['predictions_file']
                predicted=np.load(path)['probabilities']
                replay=infer(model,x[:,[union.index(d) for d in offsets]],mode,offsets)
                np.testing.assert_array_equal(replay,predicted)
                verified.append(dict(file=path.name,sha256=sha(path),replay_exact=True))
    receipt=dict(all_passed=True,real_new_frame_positions_and_sha256_checked=4304,reused_features_exact=True,
        replayed_predictions=len(verified),predictions=verified,source_sha256=sha(Path(__file__)),
        input_features_sha256=sha(directory/'input_features.npz'))
    dump(directory/'artifact_verification.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='predictions'}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    main(a.project,a.directory,a.root)
