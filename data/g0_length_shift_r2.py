"""Compute Setup numbers from the frozen real manifest, preserving official split."""
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def main():
    src=ROOT/'data/manifest/manifest_real.csv';data=src.read_bytes();df=pd.read_csv(src)
    length=df.duration_s-df.anchor_s
    assert np.allclose(length,df.post_anchor_length_s,rtol=0,atol=1e-9)
    assert dict(df.split.value_counts())=={'test':1514,'train':411,'dev':102}
    assert dict(df.split_official.value_counts())=={'test':1520,'train':507}
    assert df.decode_ok.astype(str).str.lower().eq('true').all()
    groups={}
    for col,values in [('split_official',['train','test']),('split',['train','dev','test'])]:
        for sp in values:
            a=length[df[col]==sp].to_numpy()
            groups[col+'='+sp]={'n':len(a),'q25_s':float(np.quantile(a,.25,method='linear')),
                'median_s':float(np.quantile(a,.5,method='linear')),'q75_s':float(np.quantile(a,.75,method='linear')),
                'eligible':{str(h):{'n':int((a>=h-1e-9).sum()),'fraction':float((a>=h-1e-9).mean())} for h in [4.,10.,21.5]}}
    a=groups['split_official=train'];b=groups['split_official=test']
    fractions=lambda g:', '.join(f"{100*g['eligible'][str(h)]['fraction']:.1f}%" for h in [4.,10.,21.5])
    paragraph=(f"The official IID partition of ACCIDENT exhibits a difference in post-impact observation length. "
       f"Defining this length as clip duration minus the annotated collision time, the training partition "
       f"({a['n']:,} clips) has a median of {a['median_s']:.2f} s (25th–75th percentiles: {a['q25_s']:.2f}–{a['q75_s']:.2f} s), "
       f"compared with {b['median_s']:.2f} s ({b['q25_s']:.2f}–{b['q75_s']:.2f} s) in the test partition ({b['n']:,} clips). "
       f"The proportions retaining at least 4.0, 10.0, and 21.5 s after impact are {fractions(a)} in training "
       f"and {fractions(b)} in testing, respectively. These statistics use the original IID labels, before source-video "
       f"overlap repair and development-set extraction. The observation-length difference therefore precedes our "
       f"development split. We retain the prespecified observation windows and report the eligible cohort for each window.\n")
    dest=ROOT/'report/text';dest.mkdir(parents=True,exist_ok=True)
    (dest/'g0_length_shift.md').write_text(paragraph)
    result={'source':'data/manifest/manifest_real.csv','manifest_md5':hashlib.md5(data).hexdigest(),
       'algorithm':'L=duration_s-anchor_s; check against post_anchor_length_s at abs tolerance 1e-9; group by split_official for original IID, split for repaired subsets; numpy quantile(method=linear), index (n-1)*q; eligibility L >= H-1e-9; denominator all group rows; no re-selection of H.',
       'groups':groups,'official_test_minus_train_percentage_points':{str(h):100*(b['eligible'][str(h)]['fraction']-a['eligible'][str(h)]['fraction']) for h in [4.,10.,21.5]}}
    (dest/'g0_length_shift_stats.json').write_text(json.dumps(result,indent=2)+'\n')
    assert src.read_bytes()==data
    print(json.dumps(result,indent=2));print(paragraph)

if __name__=='__main__':main()
