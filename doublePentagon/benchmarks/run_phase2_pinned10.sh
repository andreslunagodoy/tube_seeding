#!/bin/bash
# Phase 2 (parallel, L3-isolated): 10 reps x 10 triple cuts x n=5..20.
# Produces the data behind Table 3 (tab:triple-summary) and
# Figs 10-11 (fig:triple-tsolve-color, fig:triple-rss-color) of stripseeding.tex.
#
# Each cut runs as its own background process pinned to a single CPU on a
# dedicated L3 cache slice -- so the 10 batches do not share L3, and the
# cache footprint of one batch cannot evict the others'.
#
# Host: AMD EPYC 9535 (2 sockets x 64 cores x SMT2 = 256 logical CPUs).
#   L3: 32 instances of 16 MiB, each shared by one 4-core CCX (= 8 logical CPUs).
#   CCX layout: cpus {0-3,128-131} share L3#0, {4-7,132-135} share L3#1, ...
# We pick the lowest CPU in each of the first 10 L3 groups, all on NUMA0.
set -e
cd "$(dirname "$(readlink -f "$0")")"

PY="${PYTHON:-python}"

# One CPU per L3 group; the other 3 cores in each CCX are left idle so the
# pinned worker has the full 16 MiB L3 slice to itself.
CPUS=(0 4 8 12 16 20 24 28 32 36)

pids=()
for c in 1 2 3 4 5 6 7 8 9 10; do
  cpu=${CPUS[$((c - 1))]}
  out_json="triple_cuts_pinned10_c${c}.json"
  out_log="triples_pinned10_c${c}.out"
  # No early skip on existing $out_json: scan_triple_cuts.py is now resume-aware
  # (it loads $out_json, keeps records with reduced_ok=True, and re-runs only
  # the missing (n, rep) triples).
  echo "launch cut=$c cpu=$cpu (L3 group $((cpu / 4)))"
  CUT_INDEX=$c REPEATS=10 RESULTS_FILE="$out_json" \
    PYTHON_JULIACALL_THREADS=1 \
    taskset -c $cpu "$PY" scan_triple_cuts.py >> "$out_log" 2>&1 &
  pids+=($!)
done

echo "waiting on ${#pids[@]} background batches..."
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if [ $fail -eq 0 ]; then
  echo "Phase 2 done"
else
  echo "Phase 2 had failures" >&2
  exit 1
fi
