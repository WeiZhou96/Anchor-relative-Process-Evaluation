from pathlib import Path
import json,hashlib,sys
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
from ape.r2 import read_json,dump,METRICS
o=r/'outputs';ph='8ac32aae418b'
old=read_json(o/'calib_plaus'/ph/'calibration.json');ka=read_json(o/'r2_Ka/calib_plaus'/ph/'calibration.json')
pairs=read_json(o/'metrics'/ph/'_pairs.json');g4=read_json(o/'r2_Ka/gates/g4.json')
records={p.stem:read_json(p) for p in (o/'metrics'/ph).glob('*.json') if not p.name.startswith('_')}
scored=[]
for a,b in pairs['window_end_tied_pairs']:
 ra,rb=records[a],records[b]
 if any(x['family'] in ['block','trivial'] for x in [ra,rb]):continue
 same=(ra['arm_rule'],ra['arm_value'])==(rb['arm_rule'],rb['arm_value'])
 scored.append((same,-abs(ra['RMSCD']-rb['RMSCD']),a,b))
scored.sort();oldpair=list(scored[0][2:])
gh=next(x for x in g4['by_h'] if x['h_s']==10)
proofs=sorted((x['system_a'],x['system_b']) for g in gh['seed_groups'] if g['pass'] for x in g['seeds'] if x['significant'])
def describe(pair):return [dict(system_id=s,RMSCD=records[s]['RMSCD'],end_window_macro_acc=records[s]['end_window_macro_acc']) for s in pair]
payload=dict(old_H10={m:old['metrics'][m] for m in METRICS},ka_H10={m:ka['metrics'][m] for m in METRICS},old_eval_P={m:len(pairs[m]['significant_pairs']) for m in METRICS},
    old_tied_with_trivial=len(pairs['window_end_tied_pairs']),old_fig1=describe(oldpair),ka_fig1=describe(proofs[0]),
    ka_g4=[{k:v for k,v in x.items() if k not in ['pairs','seed_groups']} for x in g4['by_h']],
    old_g4='No original R2-9 gate artifact exists; first-round tied-pair demonstration is not a completed R2-9 gate.',
    sources=['ape/REPORT.md','data/REPORT.md','outputs/calib_plaus','outputs/metrics','outputs/r2_Ka/gates/g4.json','REPORT_R2_K.md'])
dump(payload,o/'r2b/first_round_baseline.json')
print(json.dumps({k:payload[k] for k in ['old_fig1','ka_fig1','old_eval_P','old_tied_with_trivial']},indent=2))

full=read_json(o/'r2b/calib_plaus'/ph/'calibration.json')
fg4=read_json(o/'r2b/gates/g4.json')
fh=next(x for x in fg4['by_h'] if x['h_s']==10)
proofs=sorted((x['system_a'],x['system_b']) for g in fh['seed_groups'] if g['pass'] for x in g['seeds'] if x['significant'])
payload['full_H10']={m:full['metrics'][m] for m in METRICS}
payload['full_g4']=[{k:v for k,v in x.items() if k not in ['pairs','seed_groups']} for x in fg4['by_h']]
payload['full_fig1']=[{k:rec[k] for k in ['system_id','RMSCD','end_window_macro_acc']} for rec in [read_json(o/'r2b/metrics'/ph/f'{s}.json') for s in proofs[0]]]
dump(payload,o/'r2b/first_round_comparison.json')
print('comparison written')
