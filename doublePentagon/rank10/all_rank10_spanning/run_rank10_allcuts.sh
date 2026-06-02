#!/bin/bash
# Scan the 5 rank-10 bottleneck-cover paths over all 11 spanning cuts, one
# (cut, path) per process, pinned to a single core (matching run_phase3_pinned10
# / the §4.7 sweep). Each run reduces all 66 rank-10 targets and reports
# Tgen/Tsolve/RSS + coverage. Override PIN= for an unpinned run.
cd "$(dirname "$(readlink -f "$0")")"
PY="${PYTHON:-python}"
PIN="${PIN-numactl --physcpubind=0 --membind=0}"
export PYTHON_JULIACALL_THREADS=1
OD="$(dirname "$(readlink -f "$0")")/outputs"
OUT="$OD/rank10_allcuts_metrics.txt"
mkdir -p "$OD"
: > "$OUT"
CUTS="3,4,7 2,5,8 2,5,7 2,4,7 1,4,6 2,4,6 3,4,8 1,5,6 3,5,8 1,5,7 1,3,6,8"
PATHS="4 16 27 42 52"      # (0,4,6) (1,5,4) (2,6,2) (4,4,2) (6,1,3)
for cut in $CUTS; do
  ctag=$(echo "$cut" | tr -d ',')
  echo "=== cut $cut ===" | tee -a "$OUT"
  for idx in $PATHS; do
    json="$OD/r10_${ctag}_p${idx}.json"
    CUT="$cut" $PIN "$PY" per_path_rss_worker_allcuts.py "$idx" "$json" 2>&1 \
      | grep -E "^METRICS|Error|Traceback|MethodError" | tee -a "$OUT"
  done
done
echo ALLDONE >> "$OUT"
