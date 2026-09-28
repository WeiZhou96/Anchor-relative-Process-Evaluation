"""Fetch the pinned author reference for the differential check; no project edits."""
from pathlib import Path
import hashlib,json,os,urllib.request
ROOT=Path(os.environ.get('APE_TMP',str(Path(__file__).resolve().parents[1]/'tmp')))/'r2/S/sources'
REV='ac104b795b8cb5fb0684c5c1afe532569361fd47'
ROOT.mkdir(parents=True,exist_ok=True)
record={}
for name in ['conditional_calibration.py','ltt.py']:
    url=f'https://raw.githubusercontent.com/liranringel/etc/{REV}/{name}'
    data=urllib.request.urlopen(url,timeout=60).read()
    target=ROOT/name
    if target.exists(): assert target.read_bytes()==data
    else: target.write_bytes(data)
    record[name]={'url':url,'sha256':hashlib.sha256(data).hexdigest()}
(ROOT/'reference_hashes.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
