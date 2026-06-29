#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the Florence-2 detection survey with the 3 attacks (fgsm / pgd / patch)
# in PARALLEL, each pinned to its own GPU and writing to its OWN output dir so
# the shared clean*.json dumps never collide.
#
# Usage (inside an srun shell on the GPU node, tmux recommended):
#   ./run_parallel_survey.sh [NUM_IMAGES] [TIER]
#   ./run_parallel_survey.sh 100            # validation run, full survey tier
#   ./run_parallel_survey.sh 5000           # full paper run
#
# Picks the 3 GPUs with the most FREE memory. Override by exporting GPUS, e.g.
#   GPUS="1 4 6" ./run_parallel_survey.sh 5000
# ---------------------------------------------------------------------------
set -u

N=${1:-100}
TIER=${2:-survey}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/.conda/envs/Capstone/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1   # stream print() to the logs live (no block-buffering)

ATTACKS=(fgsm pgd patch)

# Pick the 3 freest GPUs unless GPUS is provided.
if [ -z "${GPUS:-}" ]; then
  GPUS="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t',' -k2 -nr | head -3 | cut -d',' -f1 | tr -d ' ' | paste -sd' ')"
fi
read -r -a GPU_ARR <<< "$GPUS"
if [ "${#GPU_ARR[@]}" -lt 3 ]; then
  echo "ERROR: need 3 GPUs, got: '${GPUS}'" >&2; exit 1
fi

mkdir -p "$REPO/logs"
echo "Repo:        $REPO"
echo "Images:      $N    Tier: $TIER"
echo "GPUs:        ${GPU_ARR[0]} ${GPU_ARR[1]} ${GPU_ARR[2]}  (one attack each)"
echo "------------------------------------------------------------------"

for i in 0 1 2; do
  A="${ATTACKS[$i]}"
  G="${GPU_ARR[$i]}"
  OUT="$REPO/results_survey_florence_detection_${A}"
  LOG="$REPO/logs/survey_${A}_n${N}.log"
  nohup "$PY" "$REPO/run_survey_florence_detection.py" \
      --attacks "$A" --gpu "$G" --num-images "$N" --tier "$TIER" \
      --output-dir "$OUT" > "$LOG" 2>&1 &
  echo "[$A] gpu=$G pid=$! -> $LOG"
done

echo "------------------------------------------------------------------"
echo "Launched 3 background runs. They survive this shell (nohup)."
echo "Monitor:   tail -f $REPO/logs/survey_*_n${N}.log"
echo "GPUs:      watch -n5 nvidia-smi"
echo "Summaries: $REPO/results_survey_florence_detection_{fgsm,pgd,patch}/summary_*.json"
