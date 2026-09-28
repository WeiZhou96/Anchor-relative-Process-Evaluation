"""Read-only verification of K-b outputs and immutable source evidence."""
import os,sys,json,hashlib,subprocess,collections
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from ape.r2b import Context
from ape.r2 import read_json,dump,METRICS
from ape.r2b_identity import group_key,base_key

def main():
 c=Context(ROOT);out=c.out;checks=[]
 def check(condition,label):
  if not condition:raise AssertionError(label)
  checks.append(label)
 index=read_json(ROOT/'outputs/report_index.json')
 check(len(index['rows'])==552 and index['metrics_root']=='r2b/metrics','main index 552 rows targets K-b')
 check(set(x['system_id'] for x in index['rows'])==set(c.ids),'index identity exact')
 groups=collections.defaultdict(list)
 for s in c.nontrivial:groups[group_key(c.meta[s])].append(c.meta[s]['seed'])
 check(len(groups)==56 and all(sorted(v)==['20260903','20260904','20260905'] for v in groups.values()),'56 exact three-seed groups')
 check(len(set(base_key(c.meta[s]) for s in c.nontrivial))==9,'nine classifier bases')
 from ape import systems
 from ape.r2b_identity import metadata
 saved=(systems.arm_rule,systems.seed_of)
 def forbid_parser(*args,**kwargs):raise AssertionError('production identity reached name parser')
 systems.arm_rule=systems.seed_of=forbid_parser
 try:
  check(all(metadata(s,c.cards)==c.meta[s] for s in c.ids),'184 card identities independent of name parsers')
 finally:
  systems.arm_rule,systems.seed_of=saved

 gs=read_json(out/'gates/g3.json');g4=read_json(out/'gates/g4.json');g2=read_json(out/'gates/g2.json')
 h_summary=[]
 for h,n in zip(c.cfg.h_list_s,[1328,1113,742]):
  ph=c.cfg.pi0(h).pi_hash;p=read_json(out/'metrics'/ph/'_pairs.json')
  check(set(p['systems_real'])==set(c.real),'P exact real pool '+str(h))
  check(p['n_boot']==1000,'1000 bootstrap '+str(h))
  boot=np.load(out/'r2'/f'bootstrap_{ph}.npz')
  check(len(boot.files)==172*5 and all(boot[k].shape==(1000,) for k in boot.files),'shared bootstrap cache '+str(h))
  for m in METRICS:
   check(len(p[m]['tests'])==14706,'complete Holm family '+str(h)+' '+m)
  for s in c.ids:
   record=read_json(out/'metrics'/ph/f'{s}.json')
   check(record['N_H']==n and record['audit_split']=='test' and record['train_data_unknown']==c.meta[s]['train_data_unknown'],'eval identity/cohort '+str(h)+' '+s)
   if not s.startswith('r2__'):
    old=read_json(ROOT/'outputs/metrics'/ph/f'{s}.json')
    check(all(np.isclose(record['frozen_family'][m],old['frozen_family'][m],rtol=0,atol=1e-12) for m in METRICS),'old ref values unchanged '+str(h)+' '+s)
  for folder,num in [('calib_plaus',54),('calib',14)]:
   d=out/folder/ph;cal=read_json(d/'calibration.json');scan=pd.read_csv(d/'scan_table.csv')
   check(cal['n_systems']==184 and cal['n_pi']==num and len(scan)==184*num*5,'complete scan '+folder+' '+str(h))
   check(scan.pi_hash.nunique()==num and set(scan.system_id)==set(c.ids),'scan pool '+folder+' '+str(h))
   for m in METRICS:
    rt=pd.read_csv(d/f'R_{m.replace("@","at")}.csv')
    check(np.isclose(rt.loc[rt.pi_hash==ph,'R_M'].iloc[0],1.,rtol=0,atol=1e-12),'reference R=1 '+folder+' '+str(h)+' '+m)
    check(cal['metrics'][m]['n_pairs']==len(p[m]['significant_pairs']),'P matches calibration '+folder+' '+str(h)+' '+m)
  gh=next(x for x in g4['by_h'] if x['h_s']==h)
  for row in gh['pairs']:
   a,b=c.meta[row['system_a']],c.meta[row['system_b']]
   check(row['cross_base']==any(a[k]!=b[k] for k in ['library_round','backbone','model_kind']),'G4 base '+str(h)+' '+row['system_a']+' '+row['system_b'])
  rev=next(x for x in gs['revised_rule']['by_h'] if x['h_s']==h)
  expected=np.quantile(c.dev.loc[c.dev.post_anchor_length_s>=h-1e-9,'post_anchor_length_s'],[1/3,2/3])
  check(np.allclose(expected,rev['dev_tercile_cuts_s'],rtol=0,atol=1e-12),'G3 dev-only cuts '+str(h))
  check(all(x['own']['counts']==x['fixed']['counts'] and sum(x['own']['counts'])==n for x in rev['rows']),'G3 same cohort '+str(h))
  check(all(x['own']['n_boot_valid']==1000 and x['fixed']['n_boot_valid']==1000 for x in rev['rows']),'G3 1000 finite paired bootstrap '+str(h))
  h_summary.append(dict(h_s=h,n_eligible=n,P={m:len(p[m]['significant_pairs']) for m in METRICS},G4={k:gh[k] for k in ['n_tied','n_heterogeneous','n_cross_base_heterogeneous','n_replicated_cross_base_groups']}))
 scan=pd.read_csv(out/'r2/scan_all.csv');cells=scan.drop_duplicates(['system_id','pi_hash'])
 check(len(scan)==96*184*5,'96-point union complete')
 plaus=set(pd.read_csv(out/'calib_plaus/8ac32aae418b/scan_table.csv').pi_hash)
 ps=cells[cells.pi_hash.isin(plaus)]
 coverage=dict(n_combinations=len(cells),n_with_missing=int((cells.n_missing_lookup>0).sum()),n_missing_cells=int(cells.n_missing_lookup.sum()),max_missing_rate=float(cells.missing_lookup_rate.max()),plaus_missing_cells=int(ps.n_missing_lookup.sum()),plaus_max_missing_rate=float(ps.missing_lookup_rate.max()))
 for name in ['a1','a4','a5']:check(len(read_json(out/'ablations'/f'{name}.json')['rows'])==15,name+' all horizons/metrics')
 check(len(read_json(out/'mechanisms/p_b.json')['rows'])==15,'P-b 15 rows')
 check(len(read_json(out/'mechanisms/p_c.json')['rows'])==552,'P-c 552 rows')
 check(len(read_json(out/'gates/g5.json')['rows'])==360,'G5 360 rows')
 check(read_json(out/'gates/g5.json')['pass_gate'] is None,'G5 no invented threshold')
 check(g2['pass_gate']==next(x['pass_gate'] for x in g2['by_h'] if x['h_s']==10),'G2 reference horizon only')
 changed=subprocess.check_output(['git','diff','master','--name-only'],cwd=ROOT,text=True).splitlines()
 check(all(x.startswith(('ape/','report/','scripts/k_','tests/')) or x=='REPORT_R2_Kb.md' for x in changed),'allowed tracked changes only')
 check(subprocess.check_output(['git','rev-parse','master'],cwd=ROOT,text=True).strip().startswith('b9b143a'),'master unchanged')
 protected={}
 for name in ['protocol/pi0.yaml','protocol/pi0.frozen.sha256','protocol/pi0.frozen.json','data/manifest/manifest_real.csv']+[str(p.relative_to(ROOT)) for p in (ROOT/'prereg').glob('S1_freeze_2026-*.yaml')]:
  raw=(ROOT/name).read_bytes();base=subprocess.check_output(['git','show','master:'+name],cwd=ROOT)
  check(raw==base,'immutable source '+name);protected[name]=hashlib.sha256(raw).hexdigest()
 devhash=hashlib.sha256('\n'.join(sorted(c.dev.video_id)).encode()).hexdigest()
 check(devhash=='31ed09e84d2c12ffa37786ed1810b9d4932bb8063ee36d1497cff48252025565','dev membership unchanged')
 legacy=ROOT.parent/'legacy_outputs_hashes.json'
 if legacy.exists():
  for name,sha in read_json(legacy).items():check(hashlib.sha256((ROOT/'outputs'/name).read_bytes()).hexdigest()==sha,'legacy unchanged '+name)
 payload=dict(status='passed',n_assertions=len(checks),horizons=h_summary,coverage=coverage,protected_sha256=protected,dev_sha256=devhash,n_groups=len(groups),train_data_unknown_count=sum(c.meta[s]['train_data_unknown'] for s in c.ids),cpu_only=True)
 dump(payload,out/'verification.json');print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
