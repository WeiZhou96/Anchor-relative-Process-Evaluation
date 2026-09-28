"""Single-GPU, local Qwen pilot. No answer matrix or system-card registration."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('HF_HOME',os.path.expanduser('~/.cache/huggingface'))
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import time,json,csv,hashlib,subprocess,threading,importlib.metadata
import numpy as np
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'systems/vlm_pilot'
TMP=Path(os.environ.get('APE_TMP',str(ROOT/'tmp')))/'r2/D';LABELS=['head-on','rear-end','t-bone','sideswipe','single']

def main():
    torch.set_num_threads(4);torch.manual_seed(20260904)
    cfg=json.loads((OUT/'config.json').read_text());plan=json.loads((OUT/'prefix_audit.json').read_text())
    assert len(plan)==220 and torch.cuda.device_count()==1
    assert not (OUT/'results.jsonl').exists(),'Refuse to append to prior run; archive explicitly before restart'
    download=json.loads((TMP/'model_download.json').read_text());assert download['revision']==cfg['revision']
    start=time.monotonic();samples=[];stop=threading.Event()
    def poll():
        while not stop.is_set():
            try:
                s=subprocess.run(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                mem,util=map(int,s.stdout.strip().split(','));samples.append({'elapsed_s':time.monotonic()-start,'memory_mib':mem,'utilization_percent':util})
            except Exception:pass
            stop.wait(1)
    thread=threading.Thread(target=poll,daemon=True);thread.start()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(download['snapshot_path'],dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(download['snapshot_path'],local_files_only=True)
    torch.cuda.synchronize();load_s=time.monotonic()-start
    template=processor.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':cfg['prompt']}]}],tokenize=False,add_generation_prompt=True)
    torch.cuda.reset_peak_memory_stats();records=[]
    with (OUT/'results.jsonl').open('x') as f:
        for number,p in enumerate(plan):
            t0=time.monotonic()
            with np.load(TMP/'vlm_frames'/p['npz_name']) as stored:frames=stored['frames']
            assert hashlib.sha256(frames.tobytes()).hexdigest()==p['tensor_sha256']
            assert len(frames)==8 and max(p['frame_times_s'])<=p['end_s']
            inputs=processor(text=[template],videos=[torch.from_numpy(frames).permute(0,3,1,2)],
                do_sample_frames=False,video_metadata=[{'total_num_frames':8,'fps':p['effective_fps'],'frames_indices':list(range(8))}],
                size={'shortest_edge':28*28*4,'longest_edge':cfg['max_pixels']},return_tensors='pt').to('cuda:0')
            torch.cuda.synchronize();before=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],do_sample=False,use_cache=True)
            torch.cuda.synchronize();generation_s=time.monotonic()-before
            n_input=inputs['input_ids'].shape[1];tokens=generated[:,n_input:]
            raw=processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            rec={k:p[k] for k in ['video_id','offset_s','end_s','frame_times_s','causal_margin_s']}
            rec.update(raw_output=raw,format_compliant=raw.strip() in LABELS,pred=LABELS.index(raw.strip()) if raw.strip() in LABELS else -1,
                input_tokens=int(n_input),new_tokens=int(tokens.shape[1]),generation_s=generation_s,
                request_wall_s=time.monotonic()-t0,decode_amortized_s=p['decode_amortized_s'],
                cuda_max_allocated_mib=torch.cuda.max_memory_allocated()/2**20,cuda_max_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            f.write(json.dumps(rec)+'\n');f.flush();records.append(rec)
            del generated,inputs,tokens
            if (number+1)%11==0:print('completed',number+1,'/',len(plan),'elapsed_s',round(time.monotonic()-start,1),flush=True)
    elapsed=time.monotonic()-start;stop.set();thread.join(timeout=3)
    gen=sum(r['generation_s'] for r in records);wall=sum(r['request_wall_s'] for r in records);decode=sum(r['decode_amortized_s'] for r in records)
    mean_request=(wall+decode)/len(records)
    stats={'status':'complete','model_id':cfg['model_id'],'revision':cfg['revision'],'n_clips':20,'n_queries':len(records),
        'format_compliant_n':sum(r['format_compliant'] for r in records),'format_compliance_fraction':sum(r['format_compliant'] for r in records)/len(records),
        'format_definition':'raw_output.strip() exactly equals one of five labels; no substring/alias acceptance; no constrained decoding or retry',
        'model_load_s':load_s,'generation_sum_s':gen,'request_wall_sum_s':wall,'decode_sum_s':decode,'total_elapsed_s':elapsed,
        'generation_queries_per_s':len(records)/gen,'cached_request_queries_per_s':len(records)/wall,
        'queries_per_s_including_amortized_decode':len(records)/(wall+decode),'mean_request_with_decode_s':mean_request,
        'latency_median_s':float(np.median([r['request_wall_s'] for r in records])),
        'latency_p95_s':float(np.quantile([r['request_wall_s'] for r in records],.95)),
        'cold_first_request_s':records[0]['request_wall_s'],'warm_queries_per_s':(len(records)-1)/sum(r['request_wall_s'] for r in records[1:]),
        'cuda_max_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'cuda_max_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'nvidia_smi_peak_mib':max(s['memory_mib'] for s in samples) if samples else None,
        'zero_future_frame_violations':all(r['causal_margin_s']>=0 for r in records),
        'full_cost_estimates':{},'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','huggingface_hub','numpy','opencv-python']},
        'cost_caveat':'One model, one run, 8 frames per query; linear count extrapolation from H10-eligible dev clips, no test inference. Excludes download, one-time model loading, compressed frame cache creation and external contention. No currency price assumed.'}
    manifest=list(csv.DictReader((ROOT/'data/manifest/manifest_real.csv').open()))
    for name,rs in [('real_all',manifest),('audit_test',[r for r in manifest if r['split']=='test']),('dev',[r for r in manifest if r['split']=='dev'])]:
        rectangle=len(rs)*11;in_support=sum(sum(float(r['post_anchor_length_s'])>=d-1e-9 for d in range(11)) for r in rs)
        for scheme,count in [('rectangle_0_to_10_step1',rectangle),('within_recorded_support_0_to_10_step1',in_support)]:
            stats['full_cost_estimates'][name+'_'+scheme]={'queries':count,'estimated_gpu_hours':count*mean_request/3600}
        for step in [.5,.25]:
            count=len(rs)*(int(np.floor(21.5/step))+1)
            stats['full_cost_estimates'][name+f'_rectangle_0_to_21.5_step{step}']={'queries':count,'estimated_gpu_hours':count*mean_request/3600}
    (OUT/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    (OUT/'gpu_memory_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    print('DONE',json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()
