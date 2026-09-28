#!/usr/bin/env bash
# Install the two packages the contract allows (pytest, pyarrow) if they are
# missing. Nothing else is touched: no upgrades, no downgrades, no new channels.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}
export PY=${APE_PY}
export PYTHONPATH=${APE_ROOT}

mkdir -p ${APE_TMP}
LOG=${APE_TMP}/a_install.log
{
  echo "=== a_install $(date -Is) ==="
  MISSING=""
  for pkg in pytest pyarrow; do
    if $PY -c "import $pkg" 2>/dev/null; then
      echo "already present: $pkg"
    else
      echo "missing: $pkg"
      MISSING="$MISSING $pkg"
    fi
  done
  if [ -n "$MISSING" ]; then
    echo "installing:$MISSING"
    $PY -m pip install --no-input $MISSING
  else
    echo "nothing to install"
  fi
  echo "--- versions ---"
  $PY -c "
for m in ['numpy','pandas','scipy','yaml','pytest','pyarrow']:
    mod = __import__(m)
    print(m, getattr(mod,'__version__','?'))
"
} 2>&1 | tee "$LOG"
