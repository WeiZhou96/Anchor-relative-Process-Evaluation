"""Copy three unedited frames for the largest anchor-correspondence discrepancy."""
import hashlib
import os
import json
from pathlib import Path
import shutil

project = Path(os.environ.get('APE_MMAU_PROJECT', str(Path(__file__).resolve().parents[1] / 'work')))
base = project / 'deliverables/overlap_dense_20260920'
ledger = {r['hashcode']: r for r in map(json.loads, (project / 'deliverables/route_feasibility_20260920/outputs/planning_ledger.jsonl').read_text().splitlines())}
pairs = [json.loads(line) for line in (base / 'dense/pairs.jsonl').read_text().splitlines()]
pair = max(pairs, key=lambda r: abs(r['anchors']['b_anchor_minus_mapped_a']))
qa = base / 'raw_qa'
qa.mkdir(exist_ok=False)
items = []
for name, key, frame in [('a_published_anchor', pair['a'], pair['anchors']['anchor_a']), ('b_corresponding_to_a', pair['b'], round(pair['anchors']['mapped_a_anchor_in_b'])), ('b_published_anchor', pair['b'], pair['anchors']['anchor_b'])]:
    folder = Path(os.environ['APE_MMAU']) / ledger[key]['relative_image_directory']
    paths = [p for p in folder.iterdir() if p.stem.isdecimal() and int(p.stem) == frame]
    assert len(paths) == 1
    path = paths[0]
    target = qa / (name + path.suffix)
    shutil.copyfile(path, target)
    frames = json.loads((base / 'dense/frames' / (key + '.json')).read_text())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest == frames['raw_sha256'][frames['frames'].index(frame)]
    items.append(dict(name=name, video_id=key, frame=frame, file=target.name, source=str(path), sha256=digest))
(qa / 'manifest.json').write_text(json.dumps(dict(selection='largest absolute anchor correspondence discrepancy; diagnostic selection, not representative', pair=[pair['a'], pair['b']], native_classes=pair['native_classes'], anchors=pair['anchors'], images=items), indent=2))
print(json.dumps({'pair': [pair['a'], pair['b']], 'anchors': pair['anchors'], 'files': [r['file'] for r in items]}))
