#!/usr/bin/env bash
# Dependency check for the B track. Installs nothing: everything this track needs
# is already in the project environment. pytest is the A track's dependency, so it is only
# reported here, not installed, to avoid two jobs running pip at the same time.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

$PY -X utf8 -c "import importlib.util as u; [print(n, 'ok' if u.find_spec(n) else 'MISSING') for n in ['torch','torchvision','numpy','pandas','cv2','yaml','sklearn','scipy','pytest','pyarrow']]"
echo "b_00_deps done"
