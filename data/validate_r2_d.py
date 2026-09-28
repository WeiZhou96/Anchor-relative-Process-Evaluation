"""Validate D's artifacts and prove frozen inputs remain byte-identical to ed2b044."""
from pathlib import Path
import subprocess,csv,json,hashlib,py_compile
ROOT=Path(__file__).resolve().parents[1]
def main():
    checks={}
    for rel in ['data/manifest/manifest_real.csv','data/manifest/manifest_real_v1_official.csv','data/manifest/manifest_synthetic.csv',
                'protocol/pi0.yaml','protocol/pi0.frozen.sha256','prereg/S1_freeze_2026-09-03.yaml','prereg/S1_freeze_2026-09-03_v2.yaml']:
        expected=subprocess.run(['git','-C',str(ROOT),'show','ed2b044:'+rel],capture_output=True,check=True).stdout
        actual=(ROOT/rel).read_bytes();assert expected==actual,rel
        checks[rel]={'unchanged':True,'sha256':hashlib.sha256(actual).hexdigest()}
    with (ROOT/'data/manifest/manifest_real.csv').open() as f:real=list(csv.DictReader(f))
    with (ROOT/'data/manifest/manifest_synth.csv').open() as f:synth=list(csv.DictReader(f))
    assert list(real[0])==list(synth[0]) and len(synth)==2211
    assert len({r['video_id'] for r in synth})==2211
    assert all(r['decode_ok']=='True' and len(r['decode_hash'])==32 for r in synth)
    assert all(r['split']=='unassigned' and r['split_official']=='unassigned' for r in synth)
    assert all(0<=float(r['anchor_s'])<=float(r['duration_s']) for r in synth)
    assert all(abs(float(r['duration_s'])-float(r['anchor_s'])-float(r['post_anchor_length_s']))<1e-9 for r in synth)
    assert not {r['decode_hash'] for r in synth}&{r['decode_hash'] for r in real}
    ids=sorted(r['video_id'] for r in real if r['split']=='dev')
    assert hashlib.sha256('\n'.join(ids).encode()).hexdigest()=='31ed09e84d2c12ffa37786ed1810b9d4932bb8063ee36d1497cff48252025565'
    for p in list((ROOT/'systems/vlm_pilot').glob('*.py'))+[ROOT/'data/build_manifest_synth_r2.py',ROOT/'data/g0_length_shift_r2.py']:
        py_compile.compile(str(p),doraise=True)
    ca=json.loads((ROOT/'systems/vlm_pilot/causality_validation.json').read_text());assert ca['status']=='passed'
    results=ROOT/'systems/vlm_pilot/results.jsonl'
    status='pending'
    if (ROOT/'systems/vlm_pilot/summary.json').exists():
        rows=[json.loads(x) for x in results.read_text().splitlines()];summary=json.loads((ROOT/'systems/vlm_pilot/summary.json').read_text())
        assert len(rows)==220 and summary['n_queries']==220
        assert sum(r['format_compliant'] for r in rows)==summary['format_compliant_n']
        assert all(max(r['frame_times_s'])<=r['end_s'] for r in rows)
        assert abs(sum(r['generation_s'] for r in rows)-summary['generation_sum_s'])<1e-8
        status='passed'
    report={'status':'passed','frozen_checks':checks,'schema_columns':len(real[0]),'synthetic_rows':len(synth),
        'real_synth_md5_overlap':0,'dev_membership_unchanged':True,'causality':'passed','vlm_result_checks':status}
    (ROOT/'data/validation_r2_d.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
