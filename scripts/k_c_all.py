#!/usr/bin/env python
"""CPU-only K-c secondary-pool runner and v5 G3. See REPORT_R2_Kc.md."""
import os
import sys
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ape.r2c import run
if __name__=='__main__':run(ROOT,sys.argv[1] if len(sys.argv)>1 else 'all')
