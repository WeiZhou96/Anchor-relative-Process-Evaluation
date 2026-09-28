"""Default direct HF download of the frozen public model. Launch with nohup."""
import os
os.environ.setdefault('HF_HOME',os.path.expanduser('~/.cache/huggingface'))
from pathlib import Path
import json,time,hashlib
from huggingface_hub import snapshot_download

def main():
    here=Path(__file__).resolve().parent;tmp=Path(os.environ.get('APE_TMP',str(Path(__file__).resolve().parents[2]/'tmp')))/'r2/D'
    inv=json.loads((here/'model_inventory.json').read_text());t0=time.monotonic()
    location=Path(snapshot_download(inv['model_id'],revision=inv['revision'],allow_patterns=[f['rfilename'] for f in inv['files']],max_workers=4))
    verified=[]
    for f in inv['files']:
        path=location/f['rfilename'];assert path.stat().st_size==f['size']
        h=hashlib.sha256() if 'lfs' in f else hashlib.sha1(f"blob {f['size']}\0".encode())
        with path.open('rb') as src:
            while block:=src.read(8<<20):h.update(block)
        expected=f['lfs']['sha256'] if 'lfs' in f else f['blobId'];assert h.hexdigest()==expected
        verified.append({'filename':f['rfilename'],'size':f['size'],'digest':expected,'verified':True})
    report={'model_id':inv['model_id'],'revision':inv['revision'],'snapshot_path':str(location),'download_s':time.monotonic()-t0,
        'route':'direct Hugging Face snapshot_download','files':verified,'bytes':sum(f['size'] for f in inv['files'])}
    (tmp/'model_download.json').write_text(json.dumps(report,indent=2)+'\n')
    (here/'model_download_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
