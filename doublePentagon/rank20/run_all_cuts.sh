#!/bin/bash
# run_all_cuts.sh — run combined_solve.py on all 11 spanning cuts from Eq. (eq:spanningCutList)

CUTS=(
  "3,4,7"
  "2,5,8"
  "2,5,7"
  "2,4,7"
  "1,4,6"
  "2,4,6"
  "3,4,8"
  "1,5,6"
  "3,5,8"
  "1,5,7"
  "1,3,6,8"
)

TIMEOUT=3600 # one-hour time limit for each cut

for cut in "${CUTS[@]}"; do
    tag=$(echo "$cut" | tr -d ',')
    logfile="log_combined_cut${tag}.txt"

    echo "=============================================="
    echo "  Running combined_solve.py --cut $cut"
    echo "  Log: $logfile"
    echo "  Started at $(date)"
    echo "=============================================="

    /usr/bin/time -v -o "stats_cut${tag}.txt" \
        timeout $TIMEOUT python combined_solve.py --cut "$cut" 2>&1 | tee "$logfile"
    rc=${PIPESTATUS[0]}

    if [ $rc -eq 124 ]; then
        echo "  *** TIMED OUT after ${TIMEOUT}s ***"
    elif [ $rc -ne 0 ]; then
        echo "  *** FAILED with exit code $rc ***"
    else
        echo "  *** Completed successfully ***"
    fi
    echo
done

echo "All cuts finished at $(date)"

echo ""
echo "=== RESOURCE USAGE SUMMARY ==="
for cut in "${CUTS[@]}"; do
    tag=$(echo "$cut" | tr -d ',')
    sf="stats_cut${tag}.txt"
    if [ -f "$sf" ]; then
        wall=$(grep "Elapsed (wall clock) time" "$sf" | sed 's/.*: //')
        rss=$(grep "Maximum resident set size" "$sf" | sed 's/.*: //')
        echo "[$cut]  wall_time=$wall  max_rss=$rss"
    else
        echo "[$cut]  (no stats file)"
    fi
done
