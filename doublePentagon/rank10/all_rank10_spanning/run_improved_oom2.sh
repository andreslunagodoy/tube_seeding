#!/bin/bash
# Finish the two improved-seeding cuts that OOMed at the 179 GiB budget:
#   [1,5,6] (killed >154 GiB) and [3,5,8] (killed >162 GiB).
# Intended to run AFTER a machine restart that doubles the cgroup RAM cap
# (memory.max ~179 -> ~358 GiB). APPENDS to the existing metrics.txt and
# writes imp_156.json / imp_358.json alongside the 8 completed cuts.
#
# Guards:
#  - refuses to run if numactl missing (pinning tool, wiped on restart -> reinstall)
#  - refuses to run if the cgroup was NOT actually enlarged (would just OOM again)
#  - starts the RAM watchdog if not already running (auto-sizes to the new cap)
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY="${PYTHON:-python}"
PIN="${PIN-numactl --physcpubind=0 --membind=0}"
export PYTHON_JULIACALL_THREADS=1
OD="$(dirname "$(readlink -f "$0")")/outputs"
OUT="$OD/metrics.txt"
MONLOG="$(dirname "$(readlink -f "$0")")/outputs"/zz_monitor.log

# --- dependency guards -------------------------------------------------------
command -v numactl >/dev/null 2>&1 || {
  echo "ERROR: numactl missing (wiped on restart). Run: sudo apt-get install -y numactl" >&2; exit 2; }
"$PY" -c "import pyfeyngym" 2>/dev/null || {
  echo "ERROR: pyfeyngym/Julia env not ready. See memory julia_env_setup.md (Pkg.instantiate)." >&2; exit 3; }

# --- cgroup-size guard: need room for ~200 GiB peaks -------------------------
MAXB=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
MAXG=$(awk "BEGIN{printf \"%.0f\", $MAXB/1073741824}")
if [ "$MAXG" -lt 250 ]; then
  echo "ERROR: cgroup memory.max is only ${MAXG} GiB -- not enlarged yet." >&2
  echo "       The two cuts peak >162 GiB and need the doubled cap (~358 GiB) or they will OOM again." >&2
  exit 4
fi
echo "cgroup memory.max = ${MAXG} GiB  (OK, watchdog will kill at 85% = $(( MAXG*85/100 )) GiB)"

# --- start watchdog if not running (auto-sizes to the new cap) ---------------
if ! pgrep -f "[z]z_monitor.sh" >/dev/null 2>&1; then
  echo "starting RAM watchdog (zz_monitor.sh)..."
  nohup bash "${RAM_WATCHDOG:-/bin/true}" >/dev/null 2>&1 &
  sleep 1
fi
echo "watchdog: $(pgrep -af "[z]z_monitor.sh" | head -1)"

# --- run the two OOM cuts (APPEND, do not truncate metrics.txt) --------------
for cut in 1,5,6 3,5,8; do
  ctag=$(echo "$cut" | tr -d ',')
  echo ">>> $(date +%H:%M:%S)  starting cut $cut"
  $PIN "$PY" all66_improved_decr_allcuts.py "$OD/imp_${ctag}.json" "$cut" 2>&1 \
    | grep -E "^METRICS|Error|Traceback|MethodError" | tee -a "$OUT"
done
echo "OOM2_DONE" >> "$OUT"
echo ">>> $(date +%H:%M:%S)  both cuts attempted; results appended to $OUT"
