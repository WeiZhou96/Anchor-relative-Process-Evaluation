#!/usr/bin/env bash
# Install the only two missing packages (pytest, pyarrow) and prove nothing else moved.
# The contract forbids upgrading or downgrading anything already in the env, so the
# versions of the load-bearing packages are printed before and after.
#
# pypi.org is unreachable from this host (TLS handshake is cut), so a domestic mirror is
# tried first. If every index fails the script still exits 0 and just reports the state,
# because a missing pytest is an integration issue to escalate, not a reason to leave the
# rest of the pipeline unverified.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

echo ---- versions before
$PY -X utf8 -c "import numpy,pandas,cv2,matplotlib,scipy,sklearn,torch;print('numpy',numpy.__version__);print('pandas',pandas.__version__);print('cv2',cv2.__version__);print('matplotlib',matplotlib.__version__);print('scipy',scipy.__version__);print('sklearn',sklearn.__version__);print('torch',torch.__version__)"

echo ---- already importable?
$PY -X utf8 -c "
import importlib
for m in ('pytest','pyarrow'):
    try:
        print(m, importlib.import_module(m).__version__, 'ALREADY PRESENT')
    except Exception as e:
        print(m, 'MISSING:', type(e).__name__)
"

echo ---- try mirrors
set +e
for IDX in https://pypi.tuna.tsinghua.edu.cn/simple https://mirrors.aliyun.com/pypi/simple https://pypi.org/simple; do
  echo "trying $IDX"
  $PY -m pip install --index-url "$IDX" --timeout 20 --retries 1 -r requirements.txt
  if [ $? -eq 0 ]; then
    echo "install succeeded via $IDX"
    break
  fi
done
set -e

echo ---- versions after
$PY -X utf8 -c "import numpy,pandas,cv2,matplotlib,scipy,sklearn,torch;print('numpy',numpy.__version__);print('pandas',pandas.__version__);print('cv2',cv2.__version__);print('matplotlib',matplotlib.__version__);print('scipy',scipy.__version__);print('sklearn',sklearn.__version__);print('torch',torch.__version__)"
$PY -X utf8 -c "
import importlib
for m in ('pytest','pyarrow'):
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, 'STILL MISSING:', type(e).__name__)
"
