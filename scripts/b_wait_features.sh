#!/usr/bin/env bash
# Block until the backgrounded feature extraction exits, or until MAX_WAIT seconds
# pass. Waiting happens on the server, so the driver does not busy-poll.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"  # release: configurable paths
set -euo pipefail
cd ${APE_ROOT}

PIDFILE=${APE_TMP}/b_features.pid
LOG=${APE_TMP}/b_features.log
MAX_WAIT=${MAX_WAIT:-540}

if [ ! -f "$PIDFILE" ]; then
  echo "no pid file; job never launched"
  exit 1
fi
PID=$(cat "$PIDFILE")

WAITED=0
while kill -0 "$PID" 2>/dev/null; do
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    echo "STILL_RUNNING after ${WAITED}s"
    tail -n 2 "$LOG"
    exit 0
  fi
  sleep 15
  WAITED=$((WAITED + 15))
done

echo "PROCESS_EXITED after ${WAITED}s"
tail -n 6 "$LOG"
ls outputs/features/ | wc -l
