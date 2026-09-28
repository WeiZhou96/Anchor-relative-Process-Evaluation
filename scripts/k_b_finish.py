"""Wait for the one active calculation, then verify, compare, render and test."""
from pathlib import Path
import os,subprocess,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
LOG=Path(os.environ.get('APE_TMP',str(ROOT/'tmp')))/'r2/Kb'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT))
py=os.environ.get('APE_PY',sys.executable)
pidfile=LOG/'controls.pid'
if pidfile.exists():
 pid=int(pidfile.read_text())
 while True:
  try:os.kill(pid,0)
  except ProcessLookupError:break
  time.sleep(5)
for name,cmd in [('verify',[py,'scripts/k_b_verify.py']),('compare',[py,'scripts/k_b_compare.py']),('make_report',['make','report','ROOT='+str(ROOT)]),('tests_final',[py,'-m','pytest','-q'])]:
 with (LOG/(name+'.log')).open('wb') as log:code=subprocess.call(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 print(name,'exit',code,flush=True)
 if code:raise RuntimeError(name+' failed; see log')
(ROOT/'outputs/r2b/final_checks.json').write_text(json.dumps(dict(verify=True,compare=True,make_report=True,pytest=True),indent=2))
print('FINAL CHECKS COMPLETE',flush=True)
