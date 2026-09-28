"""Supplement real frames and rerun frozen systems with shifted input starts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pixel_pilot_20260920'))
from pixel_pilot import PrefixModel, dump, sha

OFFSETS = sorted(set(range(0, 61, 4)) | {39})
SHIFTS = [-4, 0, 4]
ARMS = [('anchor_weighted','prefix_mean','anchor'),('anchor_unweighted','prefix_mean','anchor'),
        ('prefix_mean','prefix_mean','ordered'),('mean_unweighted','prefix_mean','ordered'),
        ('gru','gru','ordered'),('gru_unweighted','gru','ordered')]


class NewFrames(Dataset):
    def __init__(self, rows, root, missing):
        self.items = []
        self.transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        for row in rows:
            folder = root / row['relative_image_directory']
            files = {int(p.stem):p for p in folder.iterdir() if p.stem.isdecimal()}
            for offset in missing:
                frame = row['anchor_frame'] + offset
                assert row['first_frame'] <= frame <= row['last_frame']
                self.items.append(files[frame])
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        raw = self.items[index].read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, hashlib.sha256(raw).hexdigest()


def infer(model, data, mode):
    output = []
    with torch.inference_mode():
        for batch in data.split(128):
            # Preserve original CUDA batch/time dimensions for exact zero-shift replay.
            padded = torch.zeros((128, 19, 512), device='cuda')
            padded[:len(batch), :len(OFFSETS)] = batch
            if mode == 'anchor':
                logits = model(padded[:, :1]).expand(-1, len(OFFSETS), -1)
            else:
                logits = model(padded)[:, :len(OFFSETS)]
            output.append(logits[:len(batch)].softmax(-1).cpu())
    return torch.cat(output).numpy()


def main(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=False)
    parent = args.project / 'deliverables/pixel_pilot_20260920/model_run'
    controls = args.project / 'deliverables/control_pilot_20260920/model_run'
    records = [json.loads(line) for line in (parent / 'cohort.jsonl').read_text().splitlines()]
    dev_indices = [i for i,r in enumerate(records) if r['split'] == 'dev']
    rows = [records[i] for i in dev_indices if records[i]['anchor_frame'] - 4 >= records[i]['first_frame'] and records[i]['anchor_frame'] + 64 <= records[i]['last_frame']]
    assert len(rows) == len(dev_indices) == 269
    union = sorted({d + s for d in OFFSETS for s in SHIFTS})
    old_plan = json.loads((parent / 'pilot_plan.json').read_text())
    old_offsets = old_plan['offsets_frames']
    missing = [d for d in union if d not in old_offsets]
    assert missing == [-4, 35, 43]
    checkpoint_paths = [(parent if arm in ('prefix_mean','gru') else controls) / f'{arm}_{seed}.pt' for arm,_,_ in ARMS for seed in old_plan['seeds']]
    plan = dict(shifts_frames=SHIFTS, horizon_frames=60, offsets=OFFSETS, union_offsets=union, new_offsets=missing,
                common_dev_n=len(rows), dropped_dev_ids=[], seeds=old_plan['seeds'], arms=ARMS,
                operation='shift actual model input start and endpoint together; labels and checkpoints fixed; evaluate relative to shifted start',
                not_retraining=True, not_correction_of_ground_truth=True, source_identity_certified=False, formal=False,
                input_step_unchanged='4-frame cadence plus offset39; shifted window H60',
                zero_shift_reference='same original checkpoints and old real-feature prefix through60; require no argmax changes',
                padding='discarded zeros beyond relative H60, shape128x19 for CUDA replay; causal outputs do not read padding',
                parent_feature_sha256=sha(parent/'features.npz'), normalizer_sha256=sha(parent/'normalizer.npz'),
                pretrained_weights_sha256=sha(args.weights), source_sha256=sha(Path(__file__)),
                checkpoint_sha256={p.name:sha(p) for p in checkpoint_paths})
    dump(args.out/'plan.json', plan)
    (args.out/'cohort.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cached = np.load(parent/'features.npz')
    assert np.array_equal(cached['video_ids'][dev_indices], [r['video_id'] for r in rows])
    assert plan['parent_feature_sha256'] == json.loads((parent/'summary.json').read_text())['feature_cache_sha256']
    dataset = NewFrames(rows, args.root, missing)
    backbone = models.resnet18(weights=None)
    backbone.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    backbone.fc = nn.Identity()
    backbone = backbone.cuda().eval()
    chunks, hashes = [], []
    with torch.inference_mode():
        for tensors, digests in DataLoader(dataset, batch_size=128, num_workers=4, shuffle=False, pin_memory=True):
            chunks.append(backbone(tensors.cuda()).cpu().numpy())
            hashes.extend(digests)
    new = np.concatenate(chunks).reshape(len(rows),len(missing),512)
    values = np.stack([cached['features'][dev_indices, old_offsets.index(d)] if d in old_offsets else new[:,missing.index(d)] for d in union], axis=1)
    np.savez_compressed(args.out/'shift_features.npz', features=values, offsets=union, video_ids=np.array([r['video_id'] for r in rows]))
    (args.out/'new_image_inputs.jsonl').write_text(''.join(json.dumps(dict(path=str(path.relative_to(args.root)), sha256=digest))+'\n' for path,digest in zip(dataset.items,hashes)))
    del backbone
    norm = np.load(parent/'normalizer.npz')
    x = torch.from_numpy((values - norm['mean'])/norm['std']).cuda()
    results = []
    for arm, kind, mode in ARMS:
        source = parent if arm in ('prefix_mean','gru') else controls
        for seed in plan['seeds']:
            model = PrefixModel(kind).cuda().eval()
            model.load_state_dict(torch.load(source/f'{arm}_{seed}.pt', map_location='cpu', weights_only=True))
            # Explicitly verify that changing discarded padding never alters visible predictions.
            check = torch.zeros((128,19,512),device='cuda')
            check[:len(OFFSETS),:len(OFFSETS)] = x[:len(OFFSETS),[union.index(d) for d in OFFSETS]]
            altered = check.clone()
            altered[:,len(OFFSETS):] = 100
            with torch.inference_mode():
                torch.testing.assert_close(model(check)[:,:len(OFFSETS)],model(altered)[:,:len(OFFSETS)],atol=0,rtol=0)
            for shift in SHIFTS:
                positions = [union.index(d + shift) for d in OFFSETS]
                probabilities = infer(model,x[:,positions],mode)
                tag = f'{arm}_{seed}_shift_{shift:+d}'
                np.savez_compressed(args.out/f'{tag}_predictions.npz', probabilities=probabilities, offsets=OFFSETS, video_ids=np.array([r['video_id'] for r in rows]), shift=shift)
                result = dict(system=f'{arm}_{seed}', arm=arm, seed=seed, shift=shift, predictions_file=f'{tag}_predictions.npz')
                if shift == 0:
                    reference = np.load(source/f'{arm}_{seed}_predictions.npz')['probabilities'][dev_indices][:,[old_offsets.index(d) for d in OFFSETS]]
                    result['zero_reference_max_probability_error'] = float(np.max(np.abs(reference-probabilities)))
                    result['zero_reference_argmax_differences'] = int((reference.argmax(-1)!=probabilities.argmax(-1)).sum())
                    np.testing.assert_allclose(probabilities, reference, atol=1e-6, rtol=1e-5)
                    assert result['zero_reference_argmax_differences'] == 0
                results.append(result)
            print(f'inferred {arm}_{seed} shifts -4/0/+4',flush=True)
    dump(args.out/'summary.json',dict(runs=results, common_dev_n=len(rows), new_images=len(dataset), reused_feature_positions=len(rows)*(len(union)-len(missing)), all_zero_shift_replays_passed=True, padding_causality_checks_passed=True, elapsed_seconds=time.time()-started))
    print('ANCHOR_SHIFT_COMPLETE',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
