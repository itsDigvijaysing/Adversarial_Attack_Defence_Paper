#!/usr/bin/env bash
# p4_status.sh: one-screen view of the Phase-4 queues started by p4_launch.sh.
#   bash scripts/p4_status.sh {test|full}
MODE=${1:-full}
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOGS="$REPO/logs/phase4_queue/$MODE"
echo "=== Phase-4 $MODE | $(date '+%Y-%m-%d %H:%M:%S') | tmux session p4_$MODE: $(tmux has-session -t p4_$MODE 2>/dev/null && echo running || echo NOT running)"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/GPU: /'
[ -f "$LOGS/ITERS" ] && echo "ITERS (Job 1 decision): $(cat "$LOGS/ITERS")"
echo
for q in "$LOGS"/queues/*.txt; do
  name=$(basename "$q" .txt)
  con="$LOGS/$name.console.log"
  [ -f "$con" ] || { echo "[$name] not started"; continue; }
  last=$(grep -E '^\[queue ' "$con" | tail -1 | cut -c1-170)
  beat=$(grep -E '\] [0-9]+/[0-9]+ \| ' "$con" | tail -1 | cut -c1-90)
  echo "[$name] $last"
  [ -n "$beat" ] && echo "      heartbeat: $beat"
done
echo
fails=$(awk -F'\t' '$3=="END" && $4!="0"' "$LOGS/status.tsv" 2>/dev/null | wc -l)
done_n=$(ls "$LOGS/done" 2>/dev/null | wc -l)
echo "finished jobs: $done_n | failed attempts: $fails"
[ "$fails" -gt 0 ] && awk -F'\t' '$3=="END" && $4!="0" {print "  FAIL " $1 " " $2 " exit " $4 ": " substr($6,1,120)}' "$LOGS/status.tsv"
