"""Frozen CLIP image embeddings on the exact pre-existing R18 frame grid."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ.setdefault('HF_HOME',os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT','40')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import json, time, hashlib
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from systems import common as C


class ExactFrames(Dataset):
    def __init__(self,rows,processor): self.rows,self.processor=rows,processor
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        import cv2
        from PIL import Image
        cv2.setNumThreads(1)
        r=self.rows[i]
        with np.load(Path(C.FEATURES_DIR)/(r.video_id+'.npz')) as old:
            times=old['t_s'].copy(); fps=float(old['fps_used'])
        indices=np.rint(times.astype(np.float64)*fps).astype(np.int64)
        assert np.max(np.abs(indices/fps-times))<1e-4
        assert np.all(np.diff(indices)>0)
        cap=cv2.VideoCapture(str(Path(C.DATA_ROOT)/r.path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        if not cap.isOpened(): raise RuntimeError('Cannot decode '+r.video_id)
        native_fps=float(cap.get(cv2.CAP_PROP_FPS))
        assert np.array_equal((indices/native_fps).astype(np.float32),times),r.video_id
        frames=[]; idx=0; pos=0
        try:
            while pos<len(indices):
                if not cap.grab(): raise RuntimeError('Early EOF '+r.video_id)
                if idx==indices[pos]:
                    ok,bgr=cap.retrieve()
                    if not ok: raise RuntimeError('Cannot retrieve '+r.video_id)
                    frames.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
                    pos+=1
                idx+=1
        finally: cap.release()
        if callable(getattr(self.processor,'preprocess',None)):
            pixels=self.processor(images=frames,return_tensors='pt')['pixel_values']
        else: pixels=torch.stack([self.processor(im) for im in frames])
        return r.video_id,pixels,times,fps


def first(items): return items[0]


def ready_clips(dataset,workers=24):
    """Bounded out-of-order CPU decoding; GPU inference remains single-threaded."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator=iter(range(len(dataset)));pending=set()
        def refill():
            while len(pending)<2*workers:
                try: idx=next(iterator)
                except StopIteration: break
                pending.add(executor.submit(dataset.__getitem__,idx))
        refill()
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                pending.remove(future)
                yield future.result()
            refill()


def load_clip_model():
    from transformers import CLIPVisionModelWithProjection,CLIPConfig
    # The mandated Torch 2.5 runtime needs a compatibility loader. This loader
    # accepts ONLY the exact published checkpoint below; SHA verification comes
    # before tensor loading. It does not start Hub auto-conversion threads.
    from huggingface_hub import hf_hub_download
    rev='57c216476eefef5ab752ec549e440a49ae4ae5f3'
    path=Path(hf_hub_download('openai/clip-vit-base-patch16','pytorch_model.bin',revision=rev))
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4<<20),b''): digest.update(chunk)
    assert digest.hexdigest()=='ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0'
    weights=torch.load(path,map_location='cpu',weights_only=True)
    cfg=CLIPConfig.from_pretrained('openai/clip-vit-base-patch16',revision=rev)
    net=CLIPVisionModelWithProjection(cfg.vision_config)
    vision={k:v for k,v in weights.items() if k.startswith(('vision_model.','visual_projection.'))}
    unexpected=set(vision)-set(net.state_dict())
    assert unexpected<= {'vision_model.embeddings.position_ids'},unexpected
    if 'vision_model.embeddings.position_ids' in unexpected:
        saved=vision.pop('vision_model.embeddings.position_ids')
        assert torch.equal(saved,torch.arange(saved.shape[-1]).reshape(1,-1))
    net.load_state_dict(vision,strict=True);net.config._commit_hash=rev
    print('CLIP_COMPATIBILITY_LOADER: SHA256-verified official weights, weights_only=True, no version changes',flush=True)
    return net


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    torch.set_num_threads(4)
    started=time.time()
    outstate=Path(C.OUTPUTS)/'r2_S'; outstate.mkdir(exist_ok=True)
    completed=outstate/'feature_backend.json'
    if completed.exists() and json.loads(completed.read_text()).get('complete'):
        print('FEATURE_CACHE_ALREADY_COMPLETE; no extraction or metadata changes',flush=True)
        return
    from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
    backend='clipb16'; attempts=[]
    for attempt in range(2):
        try:
            model=load_clip_model()
            processor=CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch16')
            break
        except Exception as e:
            attempts.append({'attempt':attempt+1,'error':repr(e)})
            print('CLIP_LOAD_FAILURE',attempt+1,repr(e),flush=True)
            if attempt==1:
                import torchvision
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                model=torchvision.models.resnet50(weights=weights)
                model.fc=torch.nn.Identity();processor=weights.transforms();backend='r50'
    model.eval().requires_grad_(False).cuda()
    out=Path(C.FEATURES_DIR)/('clip_b16' if backend=='clipb16' else 'r50')
    out.mkdir(exist_ok=True)
    state={'backend':backend,'features_dir':str(out),'dim':512 if backend=='clipb16' else 2048,
           'pretrained':True,'train_data_unknown':backend=='clipb16','load_failures':attempts,
           'model_revision':getattr(getattr(model,'config',None),'_commit_hash',None),
           'gpu_visible':os.environ['CUDA_VISIBLE_DEVICES']}
    from systems.r2_models import check_freeze
    state['freeze_evidence']=check_freeze()
    (outstate/'feature_backend.json').write_text(json.dumps(state,indent=2))
    man=C.load_manifest().sort_values('video_id')
    rows=[r for r in man.itertuples(index=False) if not (out/(r.video_id+'.npz')).exists()]
    dl=ready_clips(ExactFrames(rows,processor),workers=24)
    print('FEATURE_START',json.dumps(state),'remaining',len(rows),flush=True)
    nframes=0
    for n,(vid,pixels,times,fps) in enumerate(dl,1):
        chunks=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for s in range(0,len(pixels),128):
                x=pixels[s:s+128].cuda()
                z=model(pixel_values=x).image_embeds if backend=='clipb16' else model(x)
                if backend=='clipb16': z=torch.nn.functional.normalize(z.float(),dim=-1)
                chunks.append(z.float().cpu().numpy())
        feats=np.concatenate(chunks).astype(np.float16)
        assert len(feats)==len(times) and np.isfinite(feats).all()
        dest=out/(vid+'.npz'); tmp=dest.with_suffix('.npz.partial')
        with tmp.open('wb') as f:
            np.savez(f,t_s=times,feat=feats,fps_used=np.float32(fps),target_fps=np.float32(4),pretrained=np.bool_(True))
        os.replace(tmp,dest)
        nframes+=len(times)
        if n%25==0 or n==len(rows):
            print('FEATURE_PROGRESS',n,len(rows),'seconds',round(time.time()-started,1),'frames',nframes,flush=True)
    total_frames=0; digest=hashlib.sha256()
    for r in man.itertuples(index=False):
        with np.load(Path(C.FEATURES_DIR)/(r.video_id+'.npz')) as old,np.load(out/(r.video_id+'.npz')) as new:
            assert np.array_equal(old['t_s'],new['t_s'])
            assert len(old['feat'])==len(new['feat']) and new['feat'].shape[1]==state['dim']
            total_frames+=len(new['feat']);digest.update(r.video_id.encode());digest.update(new['t_s'].tobytes())
    state.update(n_clips=len(man),n_frames=total_frames,frame_grid_sha256=digest.hexdigest(),wall_seconds=time.time()-started,complete=True,cpu_decode_workers=24,delivery_order='as_completed',reused_completed_npz=len(man)-len(rows))
    (out/'_features_meta.json').write_text(json.dumps(state,indent=2))
    (outstate/'feature_backend.json').write_text(json.dumps(state,indent=2))
    print('FEATURE_DONE',json.dumps(state),flush=True)


if __name__=='__main__': main()
