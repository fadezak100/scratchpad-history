#!/bin/bash
# Usage: swap_watch.sh <slice array id> <h100 array id> <minutes> <indices...>
# For one hour after submission: when a whole-H100 task starts while its slice twin is still running,
# cancel the slice twin. If a slice twin finishes first, cancel the now-redundant H100 task.
# When the time is up, cancel every H100 task that never started; their slice twins keep running.
SLICE=$1; H=$2; MINUTES=$3; shift 3
END=$(( $(date +%s) + MINUTES * 60 ))
state() { squeue -h -j "$1" -o %T 2>/dev/null | head -1; }
log() { echo "$(date '+%H:%M:%S') $*"; }
log "watching slice=$SLICE h100=$H for $MINUTES min, indices: $*"
declare -A done_idx
while [ "$(date +%s)" -lt "$END" ]; do
  for i in "$@"; do
    [ -n "${done_idx[$i]}" ] && continue
    hs=$(state "${H}_$i"); ss=$(state "${SLICE}_$i")
    if [ "$hs" = "RUNNING" ]; then
      if [ "$ss" = "RUNNING" ]; then scancel "${SLICE}_$i"; log "index $i: H100 task started; cancelled slice task ${SLICE}_$i"
      else log "index $i: H100 task started; slice task already ended ($ss)"; fi
      done_idx[$i]=1
    elif [ -z "$ss" ] && [ "$hs" = "PENDING" ]; then
      scancel "${H}_$i"; log "index $i: slice task ended first; cancelled pending H100 task ${H}_$i"; done_idx[$i]=1
    fi
  done
  sleep 30
done
for i in "$@"; do
  [ -n "${done_idx[$i]}" ] && continue
  if [ "$(state "${H}_$i")" = "PENDING" ]; then scancel "${H}_$i"; log "index $i: not started within $MINUTES min; cancelled H100 task ${H}_$i, slice keeps running"
  else log "index $i: left as is (h100=$(state "${H}_$i") slice=$(state "${SLICE}_$i"))"; fi
done
log "finished"
