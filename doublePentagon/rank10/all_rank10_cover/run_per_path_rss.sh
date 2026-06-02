#!/bin/bash
# Spawn 66 fresh per-path RSS workers, parallelism limited by jobs slots.
set -e
cd "$(dirname "$(readlink -f "$0")")"

PY="${PYTHON:-python}"
PARALLEL=${PARALLEL:-10}

mkdir -p path_rss
for i in $(seq 0 65); do
  out="path_rss/p${i}.json"
  log="path_rss/p${i}.log"
  if [ -f "$out" ]; then
    continue
  fi
  $PY per_path_rss_worker.py $i $out > $log 2>&1 &
  while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do
    sleep 1
  done
done
wait
echo "all done"
