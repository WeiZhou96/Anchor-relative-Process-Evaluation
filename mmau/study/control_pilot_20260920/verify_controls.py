"""Check selected and fixed-epoch caches against histories and parent provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(base, parent, source):
    summary = json.loads((base / 'summary.json').read_text())
    assert len(summary['runs']) == 18 and summary['all_replays_exact']
    plan = json.loads((base / 'pilot_plan.json').read_text())
    assert sha(source) == plan['script_sha256']
    assert sha(parent / 'features.npz') == plan['input_feature_sha256']
    assert sha(parent / 'cohort.jsonl') == sha(base / 'cohort.jsonl')
    rows = [json.loads(line) for line in (base / 'cohort.jsonl').read_text().splitlines()]
    mask = np.array([r['split'] == 'dev' for r in rows])
    labels = np.array([r['class_code'] for r in rows])[mask]
    ids = np.array([r['video_id'] for r in rows])
    checked = []
    for run in summary['runs']:
        history = json.loads((base / (run['system'] + '_history.json')).read_text())
        for suffix, epoch in (('_predictions.npz', run['selected_epoch']), ('_epoch40.npz', 40)):
            path = base / (run['system'] + suffix)
            cache = np.load(path)
            assert np.array_equal(cache['video_ids'], ids)
            assert np.array_equal(cache['offsets'], plan['offsets_frames'])
            probs = cache['probabilities']
            assert probs.shape == (2524, 19, 9) and np.isfinite(probs).all()
            np.testing.assert_allclose(probs.sum(-1), 1, atol=1e-6)
            assert (probs >= 0).all()
            pred = probs[mask, -1].argmax(-1)
            macro = float(np.mean([(pred[labels == k] == k).mean() for k in range(9)]))
            assert abs(macro - history[epoch - 1]['dev_macro_67']) < 1e-12
            if run['arm'].startswith('anchor_'):
                assert np.all(probs.argmax(-1) == probs[:, :1].argmax(-1))
            checked.append(dict(file=path.name, sha256=sha(path), endpoint_matches_history=True))
    result = dict(all_passed=True, checked_prediction_caches=checked, selected_checkpoints_replayed_exactly=18,
                  parent_cohort_identical=True, feature_hash_matches=True, executed_source_hash_matches=True,
                  verifier_sha256=sha(Path(__file__)))
    (base / 'artifact_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_passed': True, 'prediction_caches': len(checked)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    a = p.parse_args()
    main(a.directory, a.parent, a.source)
