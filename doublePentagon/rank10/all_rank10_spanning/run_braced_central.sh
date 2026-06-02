#!/bin/bash
# Does a BRACED W=4 tube cover each central rank-10 target on the hardest prop-4
# cut (2,4,6), at low seed count / low RAM? (vs the fat rectangular W=4: ~14-18k
# seeds, ~29-36 GiB). Pinned, fresh process per target for clean per-path RSS.
cd "$(dirname "$(readlink -f "$0")")"
PY="${PYTHON:-python}"
PIN="${PIN-numactl --physcpubind=0 --membind=0}"
export PYTHON_JULIACALL_THREADS=1
OD="$(dirname "$(readlink -f "$0")")/outputs"
mkdir -p "$OD"
OUT="$OD/metrics.txt"; : > "$OUT"
CIDX=$($PY -c "import sys;sys.path.insert(0,'..');import path_coverage_rank10 as pc;print(' '.join(str(pc.ALL_TARGETS.index(t)) for t in [(2,4,4),(4,2,4),(3,3,4),(4,3,3),(3,4,3),(4,4,2)]))" 2>/dev/null)
echo "central indices: $CIDX  (cut 2,4,6, braced, W=4)" | tee -a "$OUT"
for idx in $CIDX; do
  CUT=2,4,6 $PIN "$PY" per_path_rss_worker_allcuts.py "$idx" "$OD/b_${idx}_w4.json" 4 braced 2>&1 \
    | grep -E "^METRICS|Error|Traceback|MethodError" | tee -a "$OUT"
done
echo ALLDONE >> "$OUT"
