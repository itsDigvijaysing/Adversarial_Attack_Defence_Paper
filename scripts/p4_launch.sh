#!/usr/bin/env bash
# ======================================================================
# p4_launch.sh: start every Phase-4 job in a detached tmux session, so the
# runs survive an SSH drop or the laptop going offline.
#
#   bash scripts/p4_launch.sh test    # every job of the plan on 5 images
#   bash scripts/p4_launch.sh full    # the real runs (plan below)
#   bash scripts/p4_launch.sh full --dry-run   # write the queues, start nothing
#
# Watch:   bash scripts/p4_status.sh {test|full}
# Attach:  tmux attach -t p4_full     (Ctrl-b d detaches; Ctrl-b n/p or
#          Ctrl-b <number> switches window)
# After a reboot: run the same launch command again. Finished jobs are
# skipped (logs/phase4_queue/<mode>/done/), unfinished ones resume from
# their checkpoints.
#
# This script only types the handoff's commands (docs/HANDOFF_A5000.md,
# section 4) with the numbers of the final plan; it changes no code. Every
# job runs with --deterministic (Gate 1, 17 Nov: two identical runs gave
# bit-identical checkpoints) and --gpu 0. Each window is one queue
# (scripts/p4_queue.sh); logs go to logs/phase4_queue/<mode>/ and, per job,
# to the run*.log next to its summaries.
#
# FINAL PLAN (17 Nov, deadline 20 Nov). Same first images everywhere
# (--select head), seed 1 unless the run is a repeat, PGD 50 steps of 0.002:
#   YOLO-World adaptive, 1,000 images: EnsJMG and Median x seeds 1-3,
#     Ens4W seed 1 (Table 5)
#   checks: (ii) 50 vs 100 steps on 300; (iii) eps 0.1 (alpha 0.0067) on 300;
#     (iv) mean objective and (v) identity median on 1,000
#   YOLO-World clean cost, 5,000 images, all transforms
#   Florence-2 adaptive, 1,000 images: EnsJMG and Median, seed 1
#   Florence-2 clean cost, 5,000 images, all transforms; detection and OCR
#     as two jobs (own output folders) so they run side by side
#   SmolVLM-256M, 1,000 images
# ======================================================================
set -euo pipefail
MODE=${1:?usage: p4_launch.sh test|full [--dry-run]}
DRY=${2:-}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

case "$MODE" in
  test) N_ADP=5; N_FLO=5; N_CLEAN=5; N_SMOL=5; N_CHECK=5; N_J1=5
        ROOT=results/phase4/test_small; SESSION=p4_test ;;
  full) N_ADP=1000; N_FLO=1000; N_CLEAN=5000; N_SMOL=1000; N_CHECK=1000
        N_J1=300; ROOT=results/phase4; SESSION=p4_full ;;
  *) echo "mode must be test or full"; exit 2 ;;
esac
LOGS="$REPO/logs/phase4_queue/$MODE"
Q="$LOGS/queues"
mkdir -p "$Q"

ITERS=50                       # PGD steps (check (ii) runs 100 on 300 images)
ALPHA=0.002
Y="python phase4/p4_yolo.py"
F="python phase4/p4_florence.py"
S="python phase4/p4_smolvlm.py"
D="--deterministic --gpu 0"
A="--experiment adaptive_bpda --attack-space native --pgd-select best $D"
FA="--experiment adaptive_bpda --pgd-select best $D"
RR="--results-root $ROOT"
yolo() {  # target seed [n] [log name] [extra flags]
  echo "$Y $A $RR --n ${3:-$N_ADP} --pgd-iters $ITERS --alpha $ALPHA --target $1 --seed $2 --log-name ${4:-run_$1_s$2.log} ${5:-}" | sed 's/ *$//'
}
flo() {   # target seed
  echo "$F $FA $RR --n $N_FLO --pgd-iters $ITERS --alpha $ALPHA --target $1 --seed $2 --log-name run_$1_s$2.log"
}
# Diagnostics share the summary stem of a normal run, so each gets its own
# --output-dir (CLAUDE.md, "Eps / iteration diagnostics").
J1="$Y $A --n $N_J1 --target ens_jpeg_median_gaussian --seed 1"

# --- window 1: EnsJMG x 3 seeds, then check (ii) at 100 steps -------------
{
  echo "# yolo_a: Table 5 EnsJMG, check (ii) 100 steps"
  yolo ens_jpeg_median_gaussian 1
  yolo ens_jpeg_median_gaussian 2
  yolo ens_jpeg_median_gaussian 3
  echo "$J1 --pgd-iters 100 --alpha $ALPHA --output-dir $ROOT/adaptive_bpda_yolo_iters100_n$N_J1"
} > "$Q/yolo_a.txt"

# --- window 2: Median x 3 seeds, check (v), check (ii) at 50, check (iii) --
{
  echo "# yolo_b: Table 5 Median, checks (v), (ii) 50 steps, (iii)"
  yolo median 1
  yolo median 2
  yolo median 3
  yolo median 1 "$N_CHECK" run_identity.log "--median-surrogate identity"
  echo "$J1 --pgd-iters $ITERS --alpha $ALPHA --output-dir $ROOT/adaptive_bpda_yolo_iters50_n$N_J1"
  echo "$J1 --pgd-iters $ITERS --eps 0.1 --alpha 0.0067 --output-dir $ROOT/adaptive_bpda_yolo_eps0.1"
} > "$Q/yolo_b.txt"

# --- window 3: YOLO-World clean cost, Ens4W, check (iv) -------------------
{
  echo "# yolo_c: clean cost YOLO-World, Table 5 Ens4W, check (iv)"
  echo "$Y --experiment clean_control --all-transforms $D $RR --n $N_CLEAN --seed 1"
  yolo ens_4way 1
  yolo ens_jpeg_median_gaussian 1 "$N_CHECK" run_mean.log "--eot-loss mean"
} > "$Q/yolo_c.txt"

# --- windows 4 and 5: Florence-2 adaptive, seed 1 ---------------------------
for T in ens_jpeg_median_gaussian median; do
  { echo "# flo_adaptive_$T"; flo $T 1; } > "$Q/flo_adaptive_$T.txt"
done

# --- windows 6 and 7: Florence-2 clean cost, detection and OCR -------------
# Two processes, two output folders: both write the summary stem
# clean_control_florence_alltransforms, so one folder would let the second
# _latest overwrite the first. Checkpoints are per track either way.
for TR in det ocr; do
  {
    echo "# flo_clean_$TR"
    echo "$F --experiment clean_control --all-transforms $D --tracks florence_$TR --n $N_CLEAN --seed 1 --output-dir $ROOT/clean_control_florence_$TR"
  } > "$Q/flo_clean_$TR.txt"
done

# --- window 8: the third model ------------------------------------------------
{ echo "# smolvlm"; echo "$S $D $RR --n $N_SMOL --seed 1"; } > "$Q/smolvlm.txt"

echo "[launch] mode=$MODE session=$SESSION logs=$LOGS"
for f in "$Q"/*.txt; do
  echo "---- $(basename "$f")"; grep -v '^#' "$f" | sed 's/^/    /'
done
[ "$DRY" = "--dry-run" ] && { echo "[launch] dry run: nothing started"; exit 0; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[launch] tmux session $SESSION already exists: attach with 'tmux attach -t $SESSION'"
  echo "         or kill it first ('tmux kill-session -t $SESSION') to relaunch."
  exit 1
fi
first=1
for f in "$Q"/*.txt; do
  name=$(basename "$f" .txt)
  cmd="bash $REPO/scripts/p4_queue.sh $f $LOGS; echo; echo '[window $name] queue ended: this window stays open'; exec bash"
  if [ $first = 1 ]; then
    tmux new-session -d -s "$SESSION" -n "$name" -c "$REPO" "$cmd"; first=0
  else
    tmux new-window -t "$SESSION" -n "$name" -c "$REPO" "$cmd"
  fi
  sleep 15      # stagger model loads
done
tmux new-window -t "$SESSION" -n status -c "$REPO" \
  "watch -n 60 bash $REPO/scripts/p4_status.sh $MODE"
echo "[launch] started. tmux attach -t $SESSION | bash scripts/p4_status.sh $MODE"
