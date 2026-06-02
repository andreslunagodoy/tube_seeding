#!/bin/bash
# Improved seeding (decreasing Kira-3 rule, all 66 rank-10 targets) on the 10
# triple spanning cuts, pinned, for comparison vs our path cover (Table 6).
cd "$(dirname "$(readlink -f "$0")")"
PY="${PYTHON:-python}"
PIN="${PIN-numactl --physcpubind=0 --membind=0}"
export PYTHON_JULIACALL_THREADS=1
OD="$(dirname "$(readlink -f "$0")")/outputs"
mkdir -p "$OD"; OUT="$OD/metrics.txt"; : > "$OUT"
for cut in 3,4,7 2,5,8 2,5,7 2,4,7 1,4,6 2,4,6 3,4,8 1,5,6 3,5,8 1,5,7; do
  ctag=$(echo "$cut" | tr -d ',')
  $PIN "$PY" all66_improved_decr_allcuts.py "$OD/imp_${ctag}.json" "$cut" 2>&1 \
    | grep -E "^METRICS|Error|Traceback|MethodError" | tee -a "$OUT"
done
echo ALLDONE >> "$OUT"
