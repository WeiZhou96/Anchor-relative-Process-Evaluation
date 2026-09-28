"""R2 targeted checks: causal prefixes, augmentation, and actual stopping semantics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys,types,importlib,io,contextlib
from pathlib import Path
import numpy as np
import torch
from systems import common as C
from systems.r2_models import augment
from systems.r2_stops import *


def main():
    count=0
    def check(ok,name):
        nonlocal count
        assert ok,name;count+=1;print('PASS',name,flush=True)
    rng=np.random.default_rng(193)
    p=rng.dirichlet(np.ones(5),size=(80,50));js=np.arange(-5,45)
    thr=np.linspace(.8,.5,41)
    full,com,_=apply_ringel(p,js,thr)
    for cut in [8,20,35]:
        partial,pc,_=apply_ringel(p[:,:cut],js[:cut],thr)
        check(np.array_equal(full[:,:cut],partial),'Ringel prefix causality '+str(cut))
        check(np.array_equal(com[:,:cut],pc),'Ringel flag causality '+str(cut))
    inf=np.full(41,np.inf)
    out,co,first=apply_ringel(p,js,inf)
    check(np.all(js[first]==40),'Ringel fallback fixed clock at 10 seconds')
    check(np.all(out[:,js<0]==raw_top(p)[:,js<0]),'raw pre-anchor rows')
    check(np.all(out[:,(js>=0)&(js<40)]==-1),'no early commitment fallback')
    check(not np.any(co[:,js<0]),'no pre-anchor commitments')
    q=np.zeros((2,9,5));q[:,:,2]=1
    rel=np.ones((2,9),bool);rel[0,2]=False
    ans,co,first=apply_teaser(q,np.arange(9),{},3,reliable=rel)
    check(first.tolist()==[5,2],'TEASER unreliable step resets persistence')
    q[1,1,:]=0;q[1,1,3]=1
    ans,co,first=apply_teaser(q,np.arange(9),{},3,reliable=rel)
    check(first.tolist()==[5,4],'TEASER label change resets persistence')
    q=np.full((3,9,5),.2)
    ans,co,first=apply_teaser(q,np.arange(9),{},2)
    check((ans==-1).all() and not co.any(),'TEASER never-commit rows retained')
    f=teaser_features(np.array([[[.1,.2,.4,.2,.1]]]))
    check(np.allclose(f[0,0],[2,.1,.2,.4,.2,.1,.2]),'TEASER predicted class plus probabilities plus gap')
    item={'t':np.arange(30,dtype=np.float32)*.25,'f':np.arange(90,dtype=np.float32).reshape(30,3),'anchor':2.0}
    for i in range(20):
        t,x=augment(item,np.random.default_rng(i))
        check(t[-1]==item['t'][-1] and (t>=2).any(),'augmentation retains terminal and supervised frame '+str(i))
        check(np.array_equal(x,item['f'][np.searchsorted(item['t'],t)]),'augmentation preserves original timestamps '+str(i))
    for k in [2,4]:
        keep=js%k==0
        coarse,cc,_=apply_ringel(p[:,keep],js[keep]//k,thr,delta=.25*k)
        for cut in [5,9]:
            a,c,_=apply_ringel(p[:,keep][:,:cut],(js[keep]//k)[:cut],thr,delta=.25*k)
            check(np.array_equal(a,coarse[:,:cut]),'Ringel coarse-grid causality '+str((k,cut)))
    # Direct comparison to the independently retrieved author implementation.
    sources=Path(os.environ.get('APE_TMP',str(Path(__file__).resolve().parents[1]/'tmp')))/'r2/S/sources'
    sys.path.insert(0,str(sources))
    official=importlib.import_module('conditional_calibration')
    class FrozenPosterior:
        def forward(self,x): return {'all_is_correct_estimation':x.max(-1).values,'all_scores':x}
    for n in [40,300]:
        probs=rng.dirichlet(np.ones(5),size=(n,41));labels=rng.integers(0,5,n);clusters=np.array([str(i) for i in range(n)])
        got,meta=ringel_calibrate(probs,labels,clusters,20260903)
        mask=np.array([c in set(meta['screening_clusters']) for c in clusters])
        args=types.SimpleNamespace(lambdas_step=.05,n_timesteps=41,accuracy_gap=.1,ltt_delta=.05)
        model=FrozenPosterior();x=torch.as_tensor(probs,dtype=torch.float64)
        candidate=official.candidate_screening(args,model,x[mask],labels[mask])
        with contextlib.redirect_stdout(io.StringIO()): expected=official.testing(args,model,x[~mask],labels[~mask],candidate).numpy()
        check(np.allclose(got,expected,atol=1e-6),'Ringel differential parity with author code n='+str(n))
    print('R2_LOGIC_CHECKS_OK',count,flush=True)


if __name__=='__main__': main()
