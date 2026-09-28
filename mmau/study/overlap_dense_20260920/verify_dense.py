"""Check dense traces, compare chain optimization to a reference DP, and expose gaps."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from dense_overlap import chain


def main(directory):
    generator = np.random.default_rng(9917)
    for _ in range(100):
        allowed = generator.random((12, 15)) < .2
        candidates = [dict(i=int(i), j=int(j), mae=float(generator.random())) for i, j in zip(*np.where(allowed))]
        result = chain(candidates, 15)
        dp = np.zeros((13, 16), dtype=int)
        for i in range(12):
            for j in range(15):
                dp[i + 1, j + 1] = max(dp[i, j + 1], dp[i + 1, j], dp[i, j] + int(allowed[i, j]))
        assert len(result) == dp[-1, -1]
    rows = [json.loads(line) for line in (directory / 'pairs.jsonl').read_text().splitlines()]
    assert len(rows) == 294
    details, evidence = [], []
    for row in rows:
        a, b = [json.loads((directory / 'frames' / (row[k] + '.json')).read_text()) for k in ('a', 'b')]
        matches = row['correspondences']
        assert len(matches) == row['ordered_matches']
        segments = []
        current = []
        for index, m in enumerate(matches):
            assert m['frame_a'] == a['frames'][m['i']] and m['frame_b'] == b['frames'][m['j']]
            assert m['mae'] <= .02 and m['correlation'] >= .98
            assert m['exact_bytes'] == (a['raw_sha256'][m['i']] == b['raw_sha256'][m['j']])
            assert m['exact_pixels'] == (a['pixel_sha256'][m['i']] == b['pixel_sha256'][m['j']])
            if index:
                prior = matches[index - 1]
                assert m['i'] > prior['i'] and m['j'] > prior['j']
                if m['frame_a'] - prior['frame_a'] > 5 or m['frame_b'] - prior['frame_b'] > 5:
                    segments.append(current)
                    current = []
            current.append(m)
        if current:
            segments.append(current)
        details.append(dict(a=row['a'], b=row['b'], status=row['status'], bounding_spans_are_not_gap_free_intervals=True,
                            matched_span_frame_ratio_a_over_b=(matches[-1]['frame_a'] - matches[0]['frame_a']) / (matches[-1]['frame_b'] - matches[0]['frame_b']) if len(matches) > 1 else None,
                            nearby_correspondence_runs=[dict(span_a=[s[0]['frame_a'], s[-1]['frame_a']], span_b=[s[0]['frame_b'], s[-1]['frame_b']], matched_frames=len(s)) for s in segments]))
        if row['anchors'] and row['anchors']['b_anchor_minus_mapped_a'] is not None:
            evidence.append(dict(a=row['a'], b=row['b'], status=row['status'], native_labels_differ=row['native_labels_differ'], **row['anchors']))
    supported = [r for r in rows if r['status'] == 'dense_visual_overlap_supported']
    comparable = [r for r in evidence if r['status'] == 'dense_visual_overlap_supported']
    differences = [abs(r['b_anchor_minus_mapped_a']) for r in comparable]
    ratios = [r['matched_span_frame_ratio_a_over_b'] for r in details if r['status'] == 'dense_visual_overlap_supported']
    report = dict(all_checks_passed=True, chain_random_dp_cases=100, pairs_checked=len(rows),
                  supported_pairs=len(supported), supported_anchor_comparable=len(comparable),
                  supported_anchor_abs_difference_gt1=sum(d > 1 for d in differences),
                  supported_anchor_abs_difference_gt5=sum(d > 5 for d in differences),
                  supported_anchor_abs_difference_median=float(np.median(differences)) if differences else None,
                  supported_anchor_abs_difference_max=max(differences, default=None),
                  supported_label_conflicts=sum(r['native_labels_differ'] for r in supported),
                  status_counts=dict(Counter(r['status'] for r in rows)),
                  byte_identical_pairs=sum(r['all_frames_same_bytes'] for r in rows),
                  supported_frame_span_ratio_median=float(np.median(ratios)) if ratios else None,
                  supported_frame_span_ratio_outside_0p8_1p25=sum(r < .8 or r > 1.25 for r in ratios),
                  generator_sha256=hashlib.sha256((Path(__file__).parent / 'dense_overlap.py').read_bytes()).hexdigest(),
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert report['generator_sha256'] == json.loads((directory / 'plan.json').read_text())['script_sha256']
    (directory / 'verification.json').write_text(json.dumps(report, indent=2))
    (directory / 'correspondence_runs.json').write_text(json.dumps(details, indent=2))
    (directory / 'anchor_correspondence.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    main(parser.parse_args().directory)
