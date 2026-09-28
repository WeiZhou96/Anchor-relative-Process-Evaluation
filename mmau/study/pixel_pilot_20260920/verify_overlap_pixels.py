"""Re-read original probe pixels; classify evidence without certifying sources."""
from __future__ import annotations

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    prior = args.project / 'deliverables/native_task_20260920/outputs/development_verified'
    edges = json.loads((prior/'guard_edges.json').read_text())
    groups = json.loads((prior/'guard_components.json').read_text())
    largest = max(groups,key=lambda g:len(g['members']))
    decoded = {r['hashcode']:r for r in map(json.loads,(args.project/'deliverables/decode_20260920/decode.jsonl').read_text().splitlines())}
    ids = sorted({e[k] for e in edges for k in ('a','b')})
    args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()

    def load(key: str) -> tuple[str,list[dict]]:
        r=decoded[key]
        folder=args.data_root/r['relative_image_directory']
        files={f.stem:f for f in folder.iterdir() if f.is_file()}
        results=[]
        for probe in r['probes']:
            raw=files[probe['frame']].read_bytes()
            if hashlib.blake2b(raw,digest_size=16).hexdigest()!=probe['file_digest']:
                raise ValueError(f'Original probe bytes changed: {key}/{probe["frame"]}')
            with Image.open(io.BytesIO(raw)) as im:
                rgb=im.convert('RGB')
                pix=np.asarray(rgb)
                digest=hashlib.sha256(str(pix.shape).encode()+pix.tobytes()).hexdigest()
                small=np.asarray(rgb.resize((128,72),Image.Resampling.BILINEAR),dtype=np.float32)/255
            results.append({'position':probe['position'],'dhash':probe['dhash'],
                'pixel_sha256':digest,'contrast':float(small.std()),'small':small})
        return key,results

    cache={}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i,(key,probes) in enumerate(pool.map(load,ids)):
            cache[key]=probes
            if (i+1)%400==0:print(f'probe clips {i+1}/{len(ids)}',flush=True)
    reports=[]
    for edge in edges:
        if edge['kind']!='exact_dhash_probe_candidate':continue
        a,b=cache[edge['a']],cache[edge['b']]
        matches=[]
        for x in a:
            for y in b:
                if x['dhash']!=y['dhash']:continue
                ax,ay=x['small'].reshape(-1),y['small'].reshape(-1)
                mae=float(np.abs(ax-ay).mean())
                nx,ny=ax-ax.mean(),ay-ay.mean()
                denominator=float(np.linalg.norm(nx)*np.linalg.norm(ny))
                corr=float(np.dot(nx,ny)/denominator) if denominator>1e-12 else 0.
                matches.append({'position_a':x['position'],'position_b':y['position'],
                    'dhash':x['dhash'],'rgb_mae_0_1':mae,'rgb_correlation':corr,
                    'exact_decoded_pixels':x['pixel_sha256']==y['pixel_sha256'],
                    'strong_similarity':bool(mae<=.02 and corr>=.98 and min(x['contrast'],y['contrast'])>=.02)})
        # Longest increasing one-to-one correspondence, preventing repeated/static probes from counting repeatedly.
        strong=[m for m in matches if m['strong_similarity']]
        strong.sort(key=lambda m:(m['position_a'],m['position_b']))
        lengths=[]
        for i,m in enumerate(strong):
            lengths.append(1+max([lengths[j] for j,n in enumerate(strong[:i])
                if n['position_a']<m['position_a'] and n['position_b']<m['position_b']],default=0))
        ordered=max(lengths,default=0)
        distinct=len({m['dhash'] for m in strong})
        label='strong_multiframe_similarity' if ordered>=3 and distinct>=3 else 'hash_only_or_insufficient_multiframe_evidence'
        reports.append({**edge,'evidence_class':label,'ordered_strong_matches':ordered,
            'distinct_strong_hashes':distinct,'exact_pixel_probe_pairs':sum(m['exact_decoded_pixels'] for m in matches),
            'touches_largest_guard':edge['a'] in largest['members'] or edge['b'] in largest['members'],
            'matches':matches})
    summary={'method':'Original probe byte verification, full-resolution RGB pixel digests and 128x72 RGB comparisons.',
        'thresholds_predeclared':{'max_rgb_mae':.02,'min_correlation':.98,'min_contrast':.02,'min_ordered_distinct_matches':3},
        'clips_reread':len(cache),'probe_images_reread':sum(map(len,cache.values())),
        'candidate_pairs':len(reports),'evidence_counts':dict(Counter(r['evidence_class'] for r in reports)),
        'largest_guard_members':len(largest['members']),
        'largest_guard_evidence_counts':dict(Counter(r['evidence_class'] for r in reports if r['touches_largest_guard'])),
        'retained_policy':'Keep ALL previous guard exclusions; insufficient probe agreement does not prove disjoint sources.',
        'source_identity_certified':False,'labels_or_anchors_edited':False,
        'elapsed_seconds':time.monotonic()-start,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/'pairs.json').write_text(json.dumps(reports,indent=2))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
