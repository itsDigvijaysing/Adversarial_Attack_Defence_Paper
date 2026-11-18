#!/usr/bin/env bash
# ======================================================================
# p4_queue.sh: run a list of Phase-4 commands one after another, logging
# everything. Meant to run inside tmux (scripts/p4_launch.sh starts it), so
# the jobs keep going when the SSH connection or the laptop drops.
#
#   bash scripts/p4_queue.sh <queue_file> <log_dir>
#
# queue_file: one command per line ("python phase4/..."); blank lines and
#   lines starting with # are skipped.
# log_dir: gets, per queue,
#   <queue>.console.log   every byte the jobs print (stdout + stderr),
#                         with START / END banners and exit codes
#   status.tsv            one line per START and END (shared by all queues):
#                         time, queue, event, exit code, seconds, command
#   done/<sha1>           marker of a command that exited 0. Re-running the
#                         queue (after a reboot) skips those and re-runs the
#                         rest; an unfinished job resumes from its checkpoint.
# Each job also writes its own run.log / summary files in its --output-dir
# (phase4/p4_logging.py); this console log is the raw backup of all of it.
#
# A failed job is retried once after 2 minutes (it resumes from its
# checkpoint, so a crash or an out-of-memory loses at most --checkpoint-every
# images); if it fails again it is logged and the queue moves on.
# ======================================================================
set -u
QUEUE_FILE=$(readlink -f "$1")
LOG_DIR=$(readlink -f "$2")
NAME=$(basename "$QUEUE_FILE" .txt)
REPO=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$LOG_DIR/done"
CONSOLE="$LOG_DIR/$NAME.console.log"
STATUS="$LOG_DIR/status.tsv"

# --- environment (same as docs/HANDOFF_A5000.md, section 1) -----------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vlm_ftune
cd "$REPO" || exit 1
export HF_HOME="$REPO/weights/hf"
export HF_HUB_OFFLINE=1          # weights are cached; keeps Florence-2 code fixed
export PYTHONUNBUFFERED=1        # every line reaches the logs immediately
# Up to 7 jobs share the machine: cap each one's CPU thread pool so they do
# not oversubscribe the cores (the same value in the test and the full run).
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "[queue $NAME] $(stamp) $*" | tee -a "$CONSOLE"; }

note "queue file $QUEUE_FILE | host $(hostname) | git $(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty')"
total=$(grep -cvE '^\s*(#|$)' "$QUEUE_FILE")
i=0
while IFS= read -r cmd || [ -n "$cmd" ]; do
    [[ "$cmd" =~ ^[[:space:]]*(#|$) ]] && continue
    i=$((i + 1))
    key=$(printf '%s' "$cmd" | sha1sum | cut -c1-16)
    if [ -e "$LOG_DIR/done/$key" ]; then
        note "SKIP [$i/$total] already finished: $cmd"
        continue
    fi
    for attempt in 1 2; do
        note "START [$i/$total] (attempt $attempt) $cmd"
        printf '%s\t%s\tSTART\t\t\t%s\n' "$(stamp)" "$NAME" "$cmd" >> "$STATUS"
        t0=$(date +%s)
        bash -c "$cmd" < /dev/null 2>&1 | tee -a "$CONSOLE"
        rc=${PIPESTATUS[0]}
        dt=$(( $(date +%s) - t0 ))
        printf '%s\t%s\tEND\t%s\t%s\t%s\n' "$(stamp)" "$NAME" "$rc" "$dt" "$cmd" >> "$STATUS"
        if [ "$rc" -eq 0 ]; then
            echo "$cmd" > "$LOG_DIR/done/$key"
            note "END   [$i/$total] exit 0 after $((dt / 60)) min"
            break
        fi
        note "FAIL  [$i/$total] exit $rc after $((dt / 60)) min (attempt $attempt)"
        [ "$attempt" -eq 1 ] && { note "retrying in 2 min (resumes from the checkpoint)"; sleep 120; }
    done
done < "$QUEUE_FILE"
note "QUEUE FINISHED"
