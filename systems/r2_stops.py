"""Explicit Ringel accumulated-gap and TEASER one-class adaptations (v3)."""
import numpy as np
from scipy.stats import binom
from sklearn.svm import OneClassSVM
from systems import common as C
from systems import devsel


def raw_top(probs):
    return np.where(np.isfinite(probs).all(-1),np.nan_to_num(probs,nan=-np.inf).argmax(-1),-1)


def irreversible(probs,accept,js,force_time=None,delta=0.25):
    valid=np.isfinite(probs).all(-1);top=raw_top(probs)
    fires=accept.copy()&valid
    fires[:,js<0]=False
    if force_time is not None: fires[:,np.abs(js*delta-force_time)<1e-9]|=valid[:,np.abs(js*delta-force_time)<1e-9]
    anyfire=fires.any(1)
    first=np.where(anyfire,fires.argmax(1),len(js))
    label=np.where(anyfire,top[np.arange(len(top)),first.clip(max=len(js)-1)],-1)
    committed=np.arange(len(js))[None,:]>=first[:,None]
    pred=np.where(committed,label[:,None],np.where((js<0)[None,:],top,-1))
    return pred,committed,first


def endpoint_summary(pred,y,post_len,js,delta=0.25):
    return devsel.window_summary(pred[:,js>=0],np.asarray(y),np.asarray(post_len)+1e-9,10.0,delta)


def _halts(scores,thresholds):
    fire=(scores>=thresholds)&np.isfinite(scores)
    fire[:,-1]=True
    return fire.argmax(1)


def ringel_calibrate(probs,labels,clusters,seed):
    """Algorithms in author's conditional_calibration.py; all inputs are eligible dev."""
    score=np.nanmax(probs,axis=-1);top=raw_top(probs)
    correct=top==np.asarray(labels)[:,None];late=correct[:,-1]
    unique=np.array(sorted(set(clusters)));rng=np.random.default_rng(seed);rng.shuffle(unique)
    group=set(unique[:len(unique)//2]);screen=np.array([c in group for c in clusters]);testing=~screen
    assert set(np.asarray(clusters)[screen]).isdisjoint(set(np.asarray(clusters)[testing]))
    sc=score[screen];co=correct[screen];la=late[screen];n,tmax=sc.shape
    eta=np.full(tmax,np.inf);screenlog=[]
    for t in range(tmax):
        for value in np.linspace(0,1,21):
            cand=eta.copy();cand[t]=value
            halt=_halts(sc,cand);use=halt<=t
            if not use.any(): break
            gap=la&~co[np.arange(n),halt]
            empirical=float(gap[use].mean())
            if empirical<=0.1:
                eta[t]=value
                screenlog.append({'t':t,'threshold':float(value),'n':int(use.sum()),'gap':empirical});break
    sc=score[testing];co=correct[testing];la=late[testing];n=len(sc)
    thresholds=np.full(tmax,np.inf);testlog=[]
    for t in range(tmax-1,-1,-1):
        cand=thresholds.copy();cand[t]=eta[t];stop=False
        halt=_halts(sc,cand);gap=la&~co[np.arange(n),halt]
        for upto in range(t,tmax):
            use=halt<=upto;size=int(use.sum());count=int(gap[use].sum())
            p=float(binom.cdf(count,size,0.1)) if size else 1.0
            testlog.append({'t':t,'upto':upto,'n':size,'loss_count':count,'p':p,'accepted':bool(p<=0.05 and size>0)})
            if size==0 or p>0.05: stop=True;break
        if stop: break
        thresholds=cand
    return thresholds,{'screening_n':int(screen.sum()),'testing_n':int(testing.sum()),
                       'screening_clusters':sorted(group),'testing_clusters':sorted(set(unique)-group),
                       'screening_trace':screenlog,'testing_trace':testlog,
                       'finite_early_thresholds':int(np.isfinite(thresholds[:-1]).sum()),
                       'thresholds_by_fine_post_j':[float(x) if np.isfinite(x) else None for x in thresholds],
                       'infinite_threshold_encoding':'null','accuracy_gap_alpha':0.1,'testing_delta':0.05,'lambda_grid_step':0.05,
                       'guarantee_claimed':False,'reference_endpoint_seconds':10.0}


def apply_ringel(probs,js,thresholds,delta=0.25):
    absolute=np.rint(js*delta/0.25).astype(int)
    thr=np.full(len(js),np.inf);within=(absolute>=0)&(absolute<len(thresholds))
    thr[within]=thresholds[absolute[within]]
    score=np.max(np.nan_to_num(probs,nan=-np.inf),axis=-1)
    return irreversible(probs,score>=thr,js,force_time=10.0,delta=delta)


def teaser_features(p):
    safe=np.nan_to_num(p,nan=0);top=raw_top(p)
    sort=np.sort(safe,axis=-1);margin=sort[...,-1]-sort[...,-2]
    return np.concatenate([top[...,None],safe,margin[...,None]],axis=-1)


def fit_teaser(probs,labels,post_len,js,gamma):
    x=teaser_features(probs);pred=raw_top(probs);bank={};counts={}
    for j,col in enumerate(js):
        if col<0: continue
        keep=np.isfinite(probs[:,j]).all(-1)&(pred[:,j]==labels)&(post_len+1e-9>=col*0.25)
        counts[int(col)]=int(keep.sum())
        if keep.sum()>=2:
            bank[int(col)]=OneClassSVM(kernel='rbf',nu=0.05,gamma=gamma).fit(x[keep,j])
    return bank,counts


def teaser_accept(probs,js,bank,delta=0.25):
    x=teaser_features(probs);valid=np.isfinite(probs).all(-1);reliable=np.zeros(probs.shape[:2],bool)
    absolute=np.rint(js*delta/0.25).astype(int)
    for j,col in enumerate(absolute):
        if col in bank and valid[:,j].any():
            reliable[valid[:,j],j]=bank[col].predict(x[valid[:,j],j])==1
    return reliable


def apply_teaser(probs,js,bank,v,delta=0.25,reliable=None):
    if reliable is None: reliable=teaser_accept(probs,js,bank,delta)
    top=raw_top(probs);run=np.zeros(len(probs),int);last=np.full(len(probs),-1)
    accept=np.zeros(probs.shape[:2],bool)
    for j,col in enumerate(js):
        if col<0: continue
        good=reliable[:,j]&(top[:,j]>=0)
        run=np.where(good,np.where(last==top[:,j],run+1,1),0)
        last=np.where(good,top[:,j],-1)
        accept[:,j]=run>=v
    return irreversible(probs,accept,js,delta=delta)


def select_teaser(train_probs,train_y,train_len,dev_probs,dev_y,dev_len,js):
    keep=dev_len+1e-9>=10;best=None;scan=[]
    for gamma in [1.0,10.0,100.0]:
        bank,counts=fit_teaser(train_probs,train_y,train_len,js,gamma)
        reliable=teaser_accept(dev_probs,js,bank)
        for v in [1,2,3,4,5]:
            pred,com,first=apply_teaser(dev_probs,js,bank,v,reliable=reliable)
            summ=endpoint_summary(pred,dev_y,dev_len,js)
            halt=np.where(first<len(js),np.maximum(0,js[first.clip(max=len(js)-1)]*0.25),10)
            earliness=1-float(np.minimum(halt[keep],10).mean())/10
            acc=summ['end_macro_acc'];hm=2*acc*earliness/(acc+earliness) if acc+earliness else 0
            scan.append({'gamma':gamma,'v':v,'harmonic_mean':hm,'speed':earliness,**summ})
            if best is None or hm>best[0]+1e-12: best=(hm,gamma,v,bank,counts)
    hm,gamma,v,bank,counts=best
    return bank,{'gamma':gamma,'v':v,'dev_harmonic_mean':hm,'train_fit_counts_by_fine_j':counts,'dev_scan':scan}
