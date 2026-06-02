#!/bin/bash
# Braced central tubes (3,3,4)=33, (3,4,3)=34 at W=4 on the 4 prop-4 cuts not yet
# measured, to complete the assembled-cover time/RAM table (+ verify coverage).
cd "$(dirname "$(readlink -f "$0")")"
PY="${PYTHON:-python}"
PIN="${PIN-numactl --physcpubind=0 --membind=0}"
export PYTHON_JULIACALL_THREADS=1
OD="$(dirname "$(readlink -f "$0")")/outputs"
mkdir -p "$OD"; OUT="$OD/remaining_metrics.txt"; : > "$OUT"
for cut in 2,4,7 1,4,6 3,4,7 3,4,8; do
  ctag=$(echo "$cut" | tr -d ',')
  echo "=== cut $cut (braced 33,34 @ w4) ===" | tee -a "$OUT"
  for idx in 33 34; do
    CUT="$cut" $PIN "$PY" per_path_rss_worker_allcuts.py "$idx" "$OD/b_${ctag}_${idx}_w4.json" 4 braced 2>&1 \
      | grep -E "^METRICS|Error|Traceback|MethodError" | tee -a "$OUT"
  done
done
echo ALLDONE >> "$OUT"
