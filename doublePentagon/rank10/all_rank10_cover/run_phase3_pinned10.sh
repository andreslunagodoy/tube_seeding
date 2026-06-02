#!/bin/bash
# Phase 3: 10 reps × 5 bottleneck-cover paths + 10 reps × s_max=10 ball baseline.
# All sequential, pinned to CPU 0 / NUMA 0 via numactl.
set -e
cd "$(dirname "$(readlink -f "$0")")"

PY="${PYTHON:-python}"
PIN="numactl --physcpubind=0 --membind=0"

# 5-path bottleneck cover indices (from analyze_minmax_cover.py / Table 7):
#   4=(0,4,6) 16=(1,5,4) 27=(2,6,2) 42=(4,4,2) 52=(6,1,3)
mkdir -p path_rss_pinned10
for idx in 4 16 27 42 52; do
  for rep in 0 1 2 3 4 5 6 7 8 9; do
    out="path_rss_pinned10/p${idx}_r${rep}.json"
    log="path_rss_pinned10/p${idx}_r${rep}.log"
    if [ -f "$out" ]; then
      echo "skip path=$idx rep=$rep (exists)"
      continue
    fi
    echo "=== path $idx rep $rep ==="
    $PIN $PY per_path_rss_worker.py $idx $out > $log 2>&1
  done
done

# s_max=10 ball baseline, 10 reps.
mkdir -p standard_ball_pinned10
for rep in 0 1 2 3 4 5 6 7 8 9; do
  out="standard_ball_pinned10/s10_r${rep}.json"
  log="standard_ball_pinned10/s10_r${rep}.log"
  if [ -f "$out" ]; then
    echo "skip ball rep=$rep (exists)"
    continue
  fi
  echo "=== ball s_max=10 rep $rep ==="
  $PIN $PY standard_ball_rank10_worker.py 10 $out > $log 2>&1
done

echo "Phase 3 done"
