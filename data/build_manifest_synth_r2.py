"""R2 registration-only synthetic manifest. Never writes to the dataset root."""
from pathlib import Path
import csv, hashlib, json, time, math, os
from collections import Counter
import cv2
from build_manifest import CLASS_CODES, cluster_of, md5_of

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('APE_ACCIDENT',str(ROOT/'external'/'ACCIDENT_2026')))

def main():
    started=time.monotonic()
    real=ROOT/'data/manifest/manifest_real.csv'
    frozen=real.read_bytes()
    columns=next(csv.reader(frozen.decode().splitlines()))
    source=DATA/'metadata-synthetic.csv'
    with source.open(newline='') as f: raw=list(csv.DictReader(f))
    rows=[];probes=[]
    for k,r in enumerate(raw):
        rel=r['rgb_path'];path=DATA/rel;vid=Path(rel).stem
        assert path.resolve().is_relative_to(DATA.resolve())
        dur=float(r['duration']);anchor=float(r['accident_time']);n=int(r['no_frames'])
        row=dict.fromkeys(columns,'')
        row.update(track_id='synthetic',dataset_id='ACCIDENT_synthetic',video_id=vid,path=rel,
            source_cluster_id=cluster_of(vid),native_anchor_field='accident_time',anchor_s=anchor,
            anchor_frame=int(r['accident_frame']),fps=n/dur,duration_s=dur,n_frames=n,
            post_anchor_length_s=dur-anchor,class_code=CLASS_CODES[r['type']],class_name=r['type'],
            map_status='unique',split='unassigned',split_official=r.get('split_in_distribution','unassigned'),
            split_geo=r.get('split_geo_aware',''),scene_layout='',region='',
            license_note='academic-noncommercial; see paper vs kaggle')
        note=[];ok=False;count=0;fps=0;checks={}
        try:
            row['decode_hash']=md5_of(path)
            cap=cv2.VideoCapture(str(path))
            try:
                fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for label,idx in [('first',0),('anchor',int(r['accident_frame'])),('last',count-1)]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,idx))
                    success,frame=cap.read();checks[label]=bool(success and frame is not None)
                ok=cap.isOpened() and all(checks.values())
            finally:cap.release()
        except Exception as e:note.append(type(e).__name__+': '+str(e))
        row['decode_ok']=bool(ok)
        probes.append(dict(video_id=vid,decode_ok=ok,md5_ok=bool(row['decode_hash']),cv_fps=fps,
            cv_n_frames=count,csv_fps=row['fps'],csv_n_frames=n,frame_count_matches=count==n,
            first_ok=checks.get('first',False),anchor_ok=checks.get('anchor',False),last_ok=checks.get('last',False),
            anchor_valid=0<=anchor<=dur,anchor_frame_valid=0<=int(r['accident_frame'])<n,note=';'.join(note)))
        rows.append(row)
        if (k+1)%200==0:print('processed',k+1,'elapsed_s',round(time.monotonic()-started,1),flush=True)
    assert real.read_bytes()==frozen
    out=ROOT/'data/manifest/manifest_synth.csv'
    for p,rs,cs in [(out,rows,columns),(ROOT/'data/manifest/decode_probe_synth.csv',probes,list(probes[0]))]:
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=cs,lineterminator='\n');w.writeheader();w.writerows(rs)
    summary=dict(n=len(rows),columns=columns,real_manifest_md5=hashlib.md5(frozen).hexdigest(),
        manifest_md5=md5_of(out),source_metadata_md5=md5_of(source),class_counts=dict(Counter(r['class_name'] for r in rows)),
        split_counts=dict(Counter(r['split'] for r in rows)),split_official_counts=dict(Counter(r['split_official'] for r in rows)),
        duplicate_video_ids=len(rows)-len(set(r['video_id'] for r in rows)),
        duplicate_md5_excess=sum(v-1 for k,v in Counter(r['decode_hash'] for r in rows if r['decode_hash']).items() if v>1),
        nominal_filename_groups=len(set(r['source_cluster_id'] for r in rows)),
        decode_failures=sum(not p['decode_ok'] for p in probes),md5_failures=sum(not p['md5_ok'] for p in probes),
        invalid_time_anchors=sum(not p['anchor_valid'] for p in probes),invalid_frame_anchors=sum(not p['anchor_frame_valid'] for p in probes),
        frame_count_mismatches=sum(not p['frame_count_matches'] for p in probes),
        elapsed_s=time.monotonic()-started,decode_probe='OpenCV first, annotated anchor and last frame; not full-stream validation',
        split_note='No official split columns in source CSV. Unassigned for pipeline exclusion; paper Table 2 describes all synthetic as training.',
        cluster_note='Legacy filename-prefix grouping only; no claim of independent physical events or synthetic seeds. Not used for splitting or bootstrap.',
        missing_attribute_note='Source map is a CARLA town identifier, not a geographic region or verified scene-layout label. Those manifest fields are empty; original map remains in source metadata.')
    (ROOT/'data/manifest/synth_summary_r2.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
