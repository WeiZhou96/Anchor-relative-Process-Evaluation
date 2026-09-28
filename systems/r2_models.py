"""R2 train/dev-only model selection, augmentation, and label-free inference."""
from pathlib import Path
import copy, hashlib, json, os, time
import numpy as np
import torch
from torch import nn
from systems import common as C
from systems.devsel import macro_acc
from systems.train_prefix import CausalGRUClassifier
from systems.infer_answers import clip_logits_on_grid, prefix_logits_on_grid, softmax_np

SEEDS=[20260903,20260904,20260905]
STATE=Path(C.OUTPUTS)/'r2_S'
CHECKPOINTS=Path(C.CKPT_DIR)
JS=np.arange(-11,89)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    os.replace(tmp,path)


def check_freeze():
    repo=Path(__file__).resolve().parents[1]
    ev={'commit':'153ac3fe57520f4892c9cf9214159cbbbbeca1a3','sha256':'cac139c56acc6f0db2536d8ccf5d4068de027a8c23b679e21782728318ad2704'}
    assert hashlib.sha256((repo/'prereg/S1_freeze_2026-09-04_v3.yaml').read_bytes()).hexdigest()==ev['sha256']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    return ev


def system_id(backbone,kind,seed): return f'r2__{backbone}__{kind}__seed{seed}'


def read_items(man,features_dir,cap=None):
    items=[]
    for r in man.itertuples(index=False):
        cf=C.load_features(r.video_id,str(features_dir))
        if cf is None or len(cf.feat)==0: raise RuntimeError('Missing feature: '+r.video_id)
        keep=np.ones(len(cf.t_s),bool) if cap is None else cf.t_s<=float(r.anchor_s)+cap+1e-9
        t,f=cf.t_s[keep],cf.feat[keep]
        if len(f)==0: raise RuntimeError('Empty feature: '+r.video_id)
        items.append({'video_id':r.video_id,'t':t,'f':f,'anchor':float(r.anchor_s),
                      'y':int(r.class_code) if hasattr(r,'class_code') else None,
                      'post_len':float(r.post_anchor_length_s) if hasattr(r,'post_anchor_length_s') else None})
    return items


def augment(item,rng):
    t,f=item['t'],item['f']
    pre=np.where(t<=item['anchor']+1e-9)[0]
    maxstart=float(t[pre[-1]]) if len(pre) else float(t[0])
    start=min(float(rng.uniform(0,1)),maxstart,item['anchor'])
    keep=(t>=start-1e-9)&(rng.random(len(t))>=0.1)
    keep[-1]=True
    post=np.where(t>=item['anchor']-1e-9)[0]
    if len(post) and not keep[post].any(): keep[post[-1]]=True
    return t[keep],f[keep]


def batch(items,idxs,mu,sd,device,rng=None,endpoint=False):
    seqs=[]
    for idx in idxs:
        it=items[idx]
        t,f=augment(it,rng) if rng is not None else (it['t'],it['f'])
        if endpoint:
            keep=t<=it['anchor']+10.0+1e-9;t,f=t[keep],f[keep]
        seqs.append((t,f,it))
    lengths=np.array([len(x[1]) for x in seqs])
    if (lengths==0).any(): raise RuntimeError('Empty endpoint prefix')
    x=np.zeros((len(seqs),int(lengths.max()),mu.shape[-1]),np.float32)
    sup=np.zeros(x.shape[:2],bool);y=[]
    for k,(t,f,it) in enumerate(seqs):
        x[k,:len(f)]=(f-mu)/sd
        sup[k,:len(f)]=t>=it['anchor']-1e-9
        y.append(it['y'])
    return torch.as_tensor(x,device=device),torch.as_tensor(sup,device=device),torch.as_tensor(y,device=device),lengths


def evaluate_endpoint(model,kind,items,mu,sd,device):
    eligible=[i for i,it in enumerate(items) if it['post_len']+1e-9>=10]
    pred=[];ys=[]
    model.eval()
    with torch.inference_mode():
        for s in range(0,len(eligible),32):
            ids=eligible[s:s+32]
            x,_sup,y,lens=batch(items,ids,mu,sd,device,endpoint=True)
            if kind=='mean':
                mask=torch.arange(x.shape[1],device=device)[None,:]<torch.as_tensor(lens,device=device)[:,None]
                z=(x*mask[:,:,None]).sum(1)/torch.as_tensor(lens,device=device)[:,None]
                logits=model(z)
            else:
                z,_=model(x)
                logits=z[torch.arange(len(ids),device=device),torch.as_tensor(lens-1,device=device)]
            pred.extend(logits.argmax(-1).cpu().tolist());ys.extend(y.cpu().tolist())
    return macro_acc(np.asarray(pred),np.asarray(ys))


def train_one(backbone,kind,seed,features_dir):
    freeze=check_freeze();sid=system_id(backbone,kind,seed)
    final=CHECKPOINTS/(sid+'.pt')
    if final.exists():
        ck=torch.load(final,map_location='cpu',weights_only=False)
        assert ck['freeze_commit']==freeze['commit']
        return final
    man=C.load_manifest()
    trman=man[man.split.eq('train')];dvman=man[man.split.eq('dev')]
    assert len(trman)==411 and len(dvman)==102
    cap=None if kind=='mean' else 22.0
    tr=read_items(trman,features_dir,cap);dv=read_items(dvman,features_dir,cap)
    if kind=='mean': stat=np.stack([it['f'].mean(0) for it in tr])
    else: stat=np.concatenate([it['f'] for it in tr])
    mu=stat.mean(0,keepdims=True);sd=stat.std(0,keepdims=True)+1e-6
    del stat
    ys=np.array([it['y'] for it in tr]);counts=np.bincount(ys,minlength=5)
    weights=torch.as_tensor(len(ys)/(5*counts),dtype=torch.float32,device='cuda')
    lrs=[0.001,0.01] if kind=='mean' else [0.0003,0.001]
    epochs=200 if kind=='mean' else 60;patience=30 if kind=='mean' else 10
    device=torch.device('cuda');overall=None;histories=[];started=time.time()
    for lr in lrs:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
        rng=np.random.default_rng(seed)
        model=nn.Linear(mu.shape[-1],5).cuda() if kind=='mean' else CausalGRUClassifier(mu.shape[-1],int(kind[3:])).cuda()
        opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.0001)
        best=-1;state=None;bestep=None;since=0;history=[]
        for ep in range(1,epochs+1):
            model.train();losses=[]
            order=rng.permutation(len(tr))
            for s in range(0,len(order),32):
                ids=order[s:s+32]
                x,sup,y,lens=batch(tr,ids,mu,sd,device,rng=rng)
                if kind=='mean':
                    mask=torch.arange(x.shape[1],device=device)[None,:]<torch.as_tensor(lens,device=device)[:,None]
                    z=(x*mask[:,:,None]).sum(1)/torch.as_tensor(lens,device=device)[:,None]
                    per=torch.nn.functional.cross_entropy(model(z),y,reduction='none')
                    loss=(per*weights[y]).mean()
                else:
                    logits,_=model(x);b,t,k=logits.shape
                    per=torch.nn.functional.cross_entropy(logits.reshape(-1,k),y[:,None].expand(b,t).reshape(-1),reduction='none').reshape(b,t)
                    n=sup.sum(1);valid=n>0
                    if not valid.any(): continue
                    perclip=(per*sup).sum(1)/n.clamp_min(1)
                    loss=(perclip[valid]*weights[y[valid]]).mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                losses.append(float(loss))
            score=evaluate_endpoint(model,kind,dv,mu,sd,device)
            history.append({'epoch':ep,'train_loss':float(np.mean(losses)),'dev_end_macro_acc':score})
            if score>best+1e-9:
                best=score;bestep=ep;since=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else: since+=1
            if ep==1 or ep%10==0 or since>=patience:
                print('TRAIN',sid,'lr',lr,'epoch',ep,'dev',round(score,6),'best',round(best,6),flush=True)
            if since>=patience: break
        histories.append({'lr':lr,'best_epoch':bestep,'best_dev_end_macro_acc':best,'history':history})
        # The preregistered tie break is earlier epoch, then first listed LR.
        key=(-best,bestep,lrs.index(lr))
        if overall is None or key<overall[0]: overall=(key,lr,bestep,state,best)
        del model,opt
    _,lr,ep,state,best=overall
    ck={'kind':kind,'backbone':backbone,'system_id':sid,'seed':seed,'in_dim':int(mu.shape[-1]),'state_dict':state,
        'feat_mean':mu,'feat_std':sd,'features_dir':str(features_dir),'freeze_commit':freeze['commit'],
        'hparams':{'lr':lr,'weight_decay':0.0001,'epochs_max':epochs,'patience':patience,'batch_size':32,'augmentation':'start_jitter_u0_1_drop0p1','loss':'inverse_frequency_clip_balanced_ce'},
        'train_info':{'best_epoch':ep,'best_dev_end_macro_acc':best,'wall_seconds':time.time()-started,'train_n':len(tr),'dev_n':len(dv),'train_class_counts':counts.tolist(),'candidates':histories}}
    CHECKPOINTS.mkdir(exist_ok=True)
    partial=final.with_suffix('.pt.partial');torch.save(ck,partial);os.replace(partial,final)
    write_json(STATE/'training'/(sid+'.json'),{k:v for k,v in ck.items() if k not in ['state_dict','feat_mean','feat_std']})
    print('TRAIN_DONE',sid,'seconds',round(time.time()-started,1),'best_dev',best,flush=True)
    return final


def infer(ck,view,js=JS,delta=0.25):
    """This function is passed a view without labels and never calculates scores."""
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert 'class_code' not in view.columns
    items=read_items(view,ck['features_dir'],cap=22.0)
    mu,sd=ck['feat_mean'],ck['feat_std'];kind=ck['kind']
    model=None
    if kind!='mean':
        model=CausalGRUClassifier(ck['in_dim'],int(kind[3:])).cuda()
        model.load_state_dict(ck['state_dict']);model.eval()
    probs=np.full((len(items),len(js),5),np.nan,np.float64)
    for i,it in enumerate(items):
        ends=it['anchor']+js*delta
        if kind=='mean':
            logits,ok=clip_logits_on_grid(it['f'],it['t'],ends,mu,sd,ck['state_dict']['weight'].numpy(),ck['state_dict']['bias'].numpy())
        else:
            x=torch.as_tensor((it['f']-mu)/sd,dtype=torch.float32,device='cuda')[None]
            with torch.inference_mode(): z,_=model(x)
            logits,ok=prefix_logits_on_grid(z[0].cpu().numpy(),it['t'],ends)
        probs[i,ok]=softmax_np(logits[ok])
    preds=np.where(np.isnan(probs).any(-1),-1,np.nan_to_num(probs,nan=-np.inf).argmax(-1))
    if model is not None: del model
    return np.array([it['video_id'] for it in items]),preds,probs
