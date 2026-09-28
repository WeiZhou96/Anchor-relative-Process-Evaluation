#!/usr/bin/env python
"""CPU-only recheck of V's complete report, evidence and immutable answer matrix."""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ape.r2 import dump,read_json
from ape.r2c import VLM

def validate(root=ROOT,evidence=Path(os.environ.get('APE_TMP',str(ROOT/'tmp')))/'r2/V'):
    root=Path(root);evidence=Path(evidence)
    causal=read_json(evidence/'causality_validation.json');fmt=read_json(evidence/'format_validation.json')
    path=root/'outputs/answers'/VLM/'answers.csv'
    answers=pd.read_csv(path);manifest=pd.read_csv(root/'data/manifest/manifest_real.csv')
    target=manifest[manifest.split.isin(['dev','test'])]
    per=answers.groupby('video_id').j.agg(list)
    prob=answers[[f'p{i}' for i in range(5)]].to_numpy(float);pred=answers.pred.to_numpy(int)
    expected=np.zeros_like(prob);valid=(pred>=0)&(pred<5);expected[np.flatnonzero(valid),pred[valid]]=1.
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    checks=dict(report_complete='状态：complete' in (evidence/'repo/REPORT_R2_V.md').read_text(),
        causality_passed=causal['status']=='passed' and causal['future_frame_violations']==0 and causal['future_timestamp_invariance'] and causal['all_tensor_hashes_replayed'] and causal['clips']==1616 and causal['cells']==166448,
        format_passed=fmt['status']=='passed_v4' and not fmt['v4_adapter_failures'] and fmt['exact_v4_checks'],
        n_clips=len(per),n_cells=len(answers),answer_sha256=digest,
        sha256_matches_acceptance=digest==fmt['answers_sha256'],
        exact_dev_test_ids=set(per.index)==set(target.video_id),
        all_103_unique_columns=all(sorted(js)==list(range(-8,95)) for js in per),
        exact_grid=bool((answers.delta_s==.25).all()),
        exact_probability_encoding=bool(np.array_equal(prob,expected)),
        valid_predictions=bool(np.isin(pred,[-1,0,1,2,3,4]).all()),
        zero_committed=bool((answers.committed.astype(str).str.lower().isin(['0','false'])).all()),
        split_counts=target.groupby('split').size().to_dict(),
        bot_cells=int((pred==-1).sum()),format_policy_overrides=fmt['policy_overrides'],
        legacy_unmodified_failures=fmt['legacy_unmodified_failures'])
    keys=['report_complete','causality_passed','format_passed','sha256_matches_acceptance','exact_dev_test_ids','all_103_unique_columns','exact_grid','exact_probability_encoding','valid_predictions','zero_committed']
    checks['include_vlm']=all(checks[k] for k in keys) and len(per)==1616 and len(answers)==166448
    dump(checks,root/'outputs/r2c/vlm_acceptance.json')
    print(json.dumps(checks,indent=2,ensure_ascii=False))
    if not checks['include_vlm']:raise ValueError('VLM acceptance failed; exclude from pool and run g3_primary only')
    return checks

if __name__=='__main__':validate()
