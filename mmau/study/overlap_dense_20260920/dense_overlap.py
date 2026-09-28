"""Dense correspondence for a fixed list of cross-partition visual candidates."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chain(matches, width):
    """Longest strictly increasing correspondence, then lowest accumulated error."""
    matches = sorted(matches, key=lambda m: (m['i'], m['j']))
    tree = [None] * (width + 1)
    states = []
    def better(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return a if states[a][:2] >= states[b][:2] else b
    def query(position):
        best = None
        while position > 0:
            best = better(best, tree[position])
            position -= position & -position
        return best
    start = 0
    while start < len(matches):
        end = start + 1
        while end < len(matches) and matches[end]['i'] == matches[start]['i']:
            end += 1
        updates = []
        for index in range(start, end):
            m = matches[index]
            previous = query(m['j'])
            length, quality = (0, 0) if previous is None else states[previous][:2]
            states.append((length + 1, quality - m['mae'], previous, index))
            updates.append((m['j'] + 1, len(states) - 1))
        for position, index in updates:
            while position <= width:
                tree[position] = better(tree[position], index)
                position += position & -position
        start = end
    best = query(width)
    indices = []
    while best is not None:
        indices.append(states[best][3])
        best = states[best][2]
    return [matches[i] for i in reversed(indices)]


def test_chain():
    toy = [{'i': i, 'j': j, 'mae': .01} for i, j in [(0, 0), (0, 1), (1, 1), (2, 2), (3, 0)]]
    result = chain(toy, 3)
    assert [(x['i'], x['j']) for x in result] == [(0, 0), (1, 1), (2, 2)]
    assert chain([], 10) == []
    return True


def load_clip(row, root):
    folder = root / row['relative_image_directory']
    files = sorted([p for p in folder.iterdir() if p.stem.isdecimal() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')], key=lambda p: int(p.stem))
    frames, small, coarse, digests, pixels, dhashes = [], [], [], [], [], []
    for path in files:
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert('RGB')
            array = np.asarray(rgb)
            pixels.append(hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest())
            small.append(np.asarray(rgb.resize((128, 72), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1) / 255)
            coarse.append(np.asarray(rgb.resize((16, 9), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1) / 255)
            gray = np.asarray(rgb.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
            dhashes.append(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes().hex())
        frames.append(int(path.stem))
        digests.append(hashlib.sha256(raw).hexdigest())
    return dict(frames=frames, small=np.stack(small), coarse=np.stack(coarse), raw_sha256=digests, pixel_sha256=pixels, dhash=dhashes)


def match_pair(a, b, row_a, row_b):
    x, y = a['coarse'], b['coarse']
    distances = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T
    top = min(16, len(y))
    nearest = np.argpartition(distances, top - 1, axis=1)[:, :top]
    # Add all exact-pixel correspondences, even if repeated images tie beyond top-16.
    inverse = {}
    for j, digest in enumerate(b['pixel_sha256']):
        inverse.setdefault(digest, []).append(j)
    matches = []
    for i in range(len(x)):
        js = sorted(set(nearest[i].tolist()) | set(inverse.get(a['pixel_sha256'][i], [])))
        ax = a['small'][i]
        by = b['small'][js]
        mae = np.abs(by - ax).mean(1)
        ac = ax - ax.mean()
        bc = by - by.mean(1, keepdims=True)
        denominator = np.linalg.norm(ac) * np.linalg.norm(bc, axis=1)
        corr = (bc @ ac) / np.maximum(denominator, 1e-12)
        good = (mae <= .02) & (corr >= .98) & (by.std(1) >= .02) & (ax.std() >= .02)
        for q in np.flatnonzero(good):
            j = js[q]
            matches.append(dict(i=i, j=j, frame_a=a['frames'][i], frame_b=b['frames'][j], mae=float(mae[q]), correlation=float(corr[q]), exact_pixels=a['pixel_sha256'][i] == b['pixel_sha256'][j], exact_bytes=a['raw_sha256'][i] == b['raw_sha256'][j]))
    ordered = chain(matches, len(y))
    distinct = len({a['dhash'][m['i']] for m in ordered})
    coverage = len(ordered) / min(len(x), len(y))
    supported = len(ordered) >= 10 and distinct >= 5 and coverage >= .5
    anchor = None
    if ordered:
        fa, fb = np.array([m['frame_a'] for m in ordered]), np.array([m['frame_b'] for m in ordered])
        k = int(np.searchsorted(fa, row_a['anchor_frame']))
        if k < len(fa) and fa[k] == row_a['anchor_frame']:
            mapped = float(fb[k])
            gap = 0
        elif 0 < k < len(fa) and fa[k] - fa[k - 1] <= 5:
            mapped = float(np.interp(row_a['anchor_frame'], fa[k - 1:k + 1], fb[k - 1:k + 1]))
            gap = int(fa[k] - fa[k - 1])
        else:
            mapped, gap = None, None
        anchor = dict(anchor_a=row_a['anchor_frame'], anchor_b=row_b['anchor_frame'], mapped_a_anchor_in_b=mapped, b_anchor_minus_mapped_a=None if mapped is None else row_b['anchor_frame'] - mapped, interpolation_gap_a_frames=gap, correspondence_not_anchor_ground_truth=True)
    return dict(status='dense_visual_overlap_supported' if supported else 'insufficient_dense_evidence', ordered_matches=len(ordered), shorter_clip_coverage=coverage, distinct_matched_dhashes=distinct,
                frame_counts=[len(x), len(y)], matched_span_a=[ordered[0]['frame_a'], ordered[-1]['frame_a']] if ordered else None,
                matched_span_b=[ordered[0]['frame_b'], ordered[-1]['frame_b']] if ordered else None,
                all_frames_same_bytes=a['raw_sha256'] == b['raw_sha256'],
                shared_distinct_raw_frames=len(set(a['raw_sha256']) & set(b['raw_sha256'])),
                shared_distinct_pixel_frames=len(set(a['pixel_sha256']) & set(b['pixel_sha256'])),
                native_classes=[row_a['native_class'], row_b['native_class']], native_labels_differ=row_a['native_class'] != row_b['native_class'],
                anchors=anchor, correspondences=ordered)


def main(args):
    started = time.time()
    project = args.project
    ledger_path = project / 'deliverables/route_feasibility_20260920/outputs/planning_ledger.jsonl'
    pairs_path = project / 'deliverables/pixel_pilot_20260920/overlap/pairs.json'
    ledger = {r['hashcode']: r for r in map(json.loads, ledger_path.read_text().splitlines())}
    pairs = [p for p in json.loads(pairs_path.read_text()) if p['evidence_class'] == 'strong_multiframe_similarity' and ledger[p['a']]['released_partition'] != ledger[p['b']]['released_partition']]
    assert len(pairs) == 294
    ids = [p[k] for p in pairs for k in ('a', 'b')]
    assert len(ids) == len(set(ids)) == 588
    args.out.mkdir(parents=True, exist_ok=False)
    plan = dict(candidate_pairs=len(pairs), selection='prior strong multiframe candidates crossing release partitions; fixed before dense results', matcher='16 nearest coarse RGB candidates per frame plus ALL exact full-resolution pixel matches; strictly increasing one-to-one chain maximizes length, then minimizes total MAE', thresholds=dict(mae=.02, correlation=.98, contrast=.02, min_matches=10, min_distinct_dhash=5, min_shorter_coverage=.5), limitations='heuristic visual overlap, not full source identity or absence-of-leakage certification; no negative release of guard exclusions', ledger_sha256=sha(ledger_path), candidate_sha256=sha(pairs_path), script_sha256=sha(Path(__file__)), chain_tests_passed=test_chain())
    (args.out / 'plan.json').write_text(json.dumps(plan, indent=2))
    def job(pair):
        a, b = (load_clip(ledger[pair[k]], args.root) for k in ('a', 'b'))
        result = dict(a=pair['a'], b=pair['b'], released_partitions=[ledger[pair[k]]['released_partition'] for k in ('a', 'b')], **match_pair(a, b, ledger[pair['a']], ledger[pair['b']]))
        for key, clip in ((pair['a'], a), (pair['b'], b)):
            evidence = {field: clip[field] for field in ('frames', 'raw_sha256', 'pixel_sha256', 'dhash')}
            (args.out / 'frames' / (key + '.json')).write_text(json.dumps(evidence))
        return result
    (args.out / 'frames').mkdir()
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        with (args.out / 'pairs.jsonl').open('w') as stream:
            for result in pool.map(job, pairs):
                results.append(result)
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                if len(results) % 20 == 0:
                    print(f'dense pairs {len(results)}/{len(pairs)}', flush=True)
    supported = [r for r in results if r['status'] == 'dense_visual_overlap_supported']
    summary = dict(pairs=len(results), clips=len(ids), frames_read=sum(sum(r['frame_counts']) for r in results), status_counts=dict(Counter(r['status'] for r in results)), whole_byte_identical_pairs=sum(r['all_frames_same_bytes'] for r in results), pairs_with_exact_pixel_frames=sum(r['shared_distinct_pixel_frames'] > 0 for r in results), supported_native_label_disagreements=sum(r['native_labels_differ'] for r in supported), source_identity_certified=False, exclusions_changed=False, elapsed_seconds=time.time() - started)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
