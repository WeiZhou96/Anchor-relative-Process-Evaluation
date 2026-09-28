"""Reproducible stage launcher. Run with the prescribed remote conda Python."""
from pathlib import Path
import os,sys,subprocess,argparse,json

ap=argparse.ArgumentParser()
ap.add_argument('stage',choices=['features','train','select','generate','verify','final','checks'])
args=ap.parse_args()
root=Path(__file__).resolve().parents[1]
temp=Path(os.environ.get('APE_TMP',str(root/'tmp')))/'r2/S';temp.mkdir(parents=True,exist_ok=True)
module={'features':'r2_features','train':'r2_train_all','select':'r2_select','generate':'r2_generate','verify':'r2_verify','final':'r2_final_eval','checks':'r2_checks'}[args.stage]
markers={'features':'feature_backend.json','train':'training_complete.json','select':'selection_sealed.json','generate':'generation_complete.json','verify':'verification_complete.json'}
if args.stage in markers:
    marker=root/'outputs/r2_S'/markers[args.stage]
    if marker.exists() and (args.stage!='features' or json.loads(marker.read_text()).get('complete')):
        print('Stage already complete; stored artifact:',marker);raise SystemExit(0)
env=os.environ.copy();env.update(APE_ROOT=str(root),PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES='0',HF_HOME=os.environ.get('HF_HOME',os.path.expanduser('~/.cache/huggingface')),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
pidfile=temp/(module+'.pid')
if pidfile.exists():
    pid=pidfile.read_text().strip();cmd=Path('/proc')/pid/'cmdline'
    if cmd.exists() and module.encode() in cmd.read_bytes(): raise SystemExit('Stage already active: '+pid)
log=temp/(module+'.log')
with log.open('ab') as out:
    process=subprocess.Popen(['nohup',sys.executable,'-u','-m','systems.'+module],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(process.pid))
print('PID',process.pid,'LOG',log)
