#!/usr/bin/env python
"""R2 K-a entry point; CPU only. See REPORT_R2_K.md."""
import os
from pathlib import Path
import sys
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ape.r2 import run
if __name__ == '__main__':
    run(ROOT, 'a5')
