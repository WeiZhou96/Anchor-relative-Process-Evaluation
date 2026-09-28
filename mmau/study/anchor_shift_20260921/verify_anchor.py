"""Replay all shifted-input outputs and check real-frame cache provenance."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_anchor_shift import OFFSETS, PrefixModel, infer, sha, dump


def main(project,directory):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    first=project/'deliverables/pixel_pilot_20260920/model_run'
    controls=project/'deliverables/control_pilot_20260920/model_run'
    plan=json.loads((directory/'plan.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    assert sha(Path(__file__).parent/'run_anchor_shift.py')==plan['source_sha256']
    assert sha(first/'features.npz')==plan['parent_feature_sha256']
    assert sha(first/'normalizer.npz')==plan['normalizer_sha256']
    rows=[json.loads(line) for line in (directory/'cohort.jsonl').read_text().splitlines()]
    original=[json.loads(line) for line in (first/'cohort.jsonl').read_text().splitlines()]
    assert rows==[r for r in original if r['split']=='dev']
    cache=np.load(directory/'shift_features.npz')
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
    assert len(inputs)==807
    for i,row in enumerate(rows):
        for j,offset in enumerate(plan['new_offsets']):
            item=inputs[i*3+j]
            assert str(Path(item['path']).parent)==row['relative_image_directory']
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
            for shift in plan['shifts_frames']:
                run=next(r for r in summary['runs'] if r['system']==f'{arm}_{seed}' and r['shift']==shift)
                path=directory/run['predictions_file']
                predicted=np.load(path)['probabilities']
                replay=infer(model,x[:,[union.index(d+shift) for d in OFFSETS]],mode)
                np.testing.assert_array_equal(replay,predicted)
                verified.append(dict(file=path.name,sha256=sha(path),replay_exact=True))
    receipt=dict(all_passed=True,real_new_frame_positions_checked=807,reused_features_exact=True,
        replayed_predictions=len(verified),predictions=verified,all_zero_shift_argmax_differences=sum(r.get('zero_reference_argmax_differences',0) for r in summary['runs']),
        max_zero_shift_probability_error=max(r.get('zero_reference_max_probability_error',0) for r in summary['runs']),
        source_sha256=sha(Path(__file__)),shift_features_sha256=sha(directory/'shift_features.npz'))
    dump(directory/'artifact_verification.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='predictions'}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args()
    main(a.project,a.directory)
