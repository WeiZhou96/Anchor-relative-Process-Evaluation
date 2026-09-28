"""Freeze a reproducible 20-clip dev workload and causal video tensors (CPU only)."""
from pathlib import Path
import csv, json, hashlib, time, math, os
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[2]
TMP=Path(os.environ.get('APE_TMP',str(ROOT/'tmp')))/'r2/D'
DATA=Path(os.environ.get('APE_ACCIDENT',str(ROOT/'external'/'ACCIDENT_2026')))
PROMPT='''You are shown sampled frames from one fixed roadside camera in chronological order. The footage stops at the current observation time.
Classify the collision configuration using only these frames. Choose exactly one label from: head-on, rear-end, t-bone, sideswipe, single.
Output only the chosen label, with no explanation or additional text.'''

def select_indices(times,end_s,n=8):
    """Only prior timestamps affect selection. No nearest-neighbor future rounding."""
    times=np.asarray(times,dtype=float)
    available=np.flatnonzero(times<=end_s)
    if len(available)<n:raise ValueError('Fewer than eight causal frames')
    targets=np.linspace(times[available[0]],times[available[-1]],n)
    ids=np.searchsorted(times,targets,side='right')-1
    assert len(set(ids.tolist()))==n and (times[ids]<=end_s).all()
    return ids

def main():
    t0=time.monotonic();manifest=ROOT/'data/manifest/manifest_real.csv';before=manifest.read_bytes()
    rows=list(csv.DictReader(before.decode().splitlines()))
    pool=[r for r in rows if r['split']=='dev' and float(r['post_anchor_length_s'])>=10-1e-9 and r['decode_ok']=='True']
    ranked=sorted(pool,key=lambda r:hashlib.sha256(('20260904:'+r['video_id']).encode()).hexdigest())
    selected=ranked[:20];assert len(selected)==20
    out=ROOT/'systems/vlm_pilot';out.mkdir(parents=True,exist_ok=True)
    cache=TMP/'vlm_frames';cache.mkdir(exist_ok=True)
    config={'model_id':'Qwen/Qwen2.5-VL-7B-Instruct','revision':json.loads((TMP/'model_revision.json').read_text())['revision'],
       'seed':20260904,'selection':'Sort H10-eligible dev clips by SHA256(20260904:video_id); take first 20; no label or prediction selection.',
       'eligible_dev_pool_n':len(pool),'delta_s':1.,'offsets_s':list(range(11)),'frames_per_query':8,
       'prompt':PROMPT,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
       'precision':'bfloat16','attention':'sdpa','batch_size':1,'do_sample':False,'max_new_tokens':16,
       'max_pixels':224*392,'manifest_md5':hashlib.md5(before).hexdigest(),
       'causality':'OpenCV decoded presentation timestamps; select left neighbors only within t<=anchor+offset; no full-clip metadata, labels, filenames, anchor or end time in model input; video tensor already sampled; no padding.',
       'timing_approximation':'Model receives effective constant fps over 8 selected frames; exact OpenCV timestamps retained separately. No future pixels are added.',
       'not_in_system_library':True,'train_data_unknown':True}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    with (out/'selection.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['video_id','source_cluster_id','split','anchor_s','post_anchor_length_s','class_name'],lineterminator='\n')
        w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in selected)
    records=[];decode_total=0
    for number,r in enumerate(selected):
        ts=time.monotonic();end=float(r['anchor_s'])+10
        cap=cv2.VideoCapture(str(DATA/r['path']));assert cap.isOpened()
        frames=[];times=[];idxs=[];idx=0
        while True:
            ok,bgr=cap.read()
            if not ok:break
            pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if pts>end:break
            if times and pts<=times[-1]:raise ValueError('Non-increasing decoded timestamps: '+r['video_id'])
            h,w=bgr.shape[:2];scale=min(1.,math.sqrt((224*392)/(h*w)))
            bgr=cv2.resize(bgr,(max(28,int(w*scale)),max(28,int(h*scale))),interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));times.append(pts);idxs.append(idx);idx+=1
        cap.release()
        assert times and times[-1]>end-.3,('Decoded footage does not cover H10',r['video_id'],times[-1],end)
        decode_s=time.monotonic()-ts;decode_total+=decode_s
        for offset in range(11):
            endpoint=float(r['anchor_s'])+offset;chosen=select_indices(times,endpoint)
            tensor=np.stack([frames[i] for i in chosen]);selected_times=np.asarray(times)[chosen]
            name=f'{number:02d}_{offset:02d}.npz';np.savez_compressed(cache/name,frames=tensor)
            rec={'video_id':r['video_id'],'offset_s':offset,'end_s':endpoint,'npz_name':name,
                'frame_indices':[idxs[i] for i in chosen],'frame_times_s':selected_times.tolist(),
                'max_frame_s':float(selected_times[-1]),'causal_margin_s':endpoint-float(selected_times[-1]),
                'effective_fps':7/float(selected_times[-1]-selected_times[0]),
                'timing_approximation_max_s':float(np.max(np.abs(selected_times-np.linspace(selected_times[0],selected_times[-1],8)))),
                'decode_amortized_s':decode_s/11,'tensor_sha256':hashlib.sha256(tensor.tobytes()).hexdigest()}
            records.append(rec)
        print('prepared',number+1,r['video_id'],'decode_s',round(decode_s,3),flush=True)
    (out/'prefix_audit.json').write_text(json.dumps(records,indent=2)+'\n')
    (out/'preparation.json').write_text(json.dumps({'elapsed_s':time.monotonic()-t0,'decode_s':decode_total,'queries':len(records)},indent=2)+'\n')
    assert manifest.read_bytes()==before and len(records)==220
    print('PREPARATION DONE',len(records),time.monotonic()-t0,flush=True)

if __name__=='__main__':main()
