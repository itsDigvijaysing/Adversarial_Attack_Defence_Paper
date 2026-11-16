#!/usr/bin/env bash
# One-shot setup for a fresh GPU server. Run from anywhere; safe to re-run
# (every step skips what is already there). Needs internet (run it on the
# login node of a cluster); the runs themselves then work offline.
#
#   bash scripts/setup_server.sh              # everything
#   SKIP_ENV=1 bash scripts/setup_server.sh   # data + weights only
#
# Afterwards, in every new shell / tmux pane / SLURM job:
#   conda activate vlm_ftune && cd <repo> && export HF_HOME=$PWD/weights/hf
#
# Disk: env ~7 GB, COCO val2017 ~1.6 GB (zip + images), weights ~1 GB.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-vlm_ftune}"
cd "$REPO"
export HF_HOME="$REPO/weights/hf"
echo "[setup] repo $REPO | env $ENV_NAME | HF_HOME $HF_HOME"
df -h "$REPO" | tail -1

# --- 1. conda (install Miniforge in $HOME if there is none) --------------
if ! command -v conda >/dev/null 2>&1; then
  for base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
    [ -f "$base/etc/profile.d/conda.sh" ] && source "$base/etc/profile.d/conda.sh" && break
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] no conda found: installing Miniforge into $HOME/miniforge3"
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- 2. environment ------------------------------------------------------
if [ "${SKIP_ENV:-0}" != 1 ]; then
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] env $ENV_NAME exists (delete it with conda env remove -n $ENV_NAME to rebuild)"
  else
    conda env create -n "$ENV_NAME" -f environment.yml
  fi
fi
conda activate "$ENV_NAME"
PY="$(command -v python)"

# --- 3. GPU check: torch 2.6.0+cu118 is built for sm_50..sm_90 ----------
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader || true
  if nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | awk -F. '$1 >= 10 {bad=1} END {exit !bad}'; then
    echo "[setup] WARNING: Blackwell-class GPU (sm_100+). torch 2.6.0+cu118 has no"
    echo "        kernels for it; install a cu128 build of torch instead."
  fi
else
  echo "[setup] no nvidia-smi here (login node?); GPU check skipped"
fi

# --- 4. COCO val2017 + instances_val2017.json ----------------------------
mkdir -p downloads
if [ "$(ls val2017 2>/dev/null | grep -c '\.jpg$')" != 5000 ]; then
  wget -c -P downloads http://images.cocodataset.org/zips/val2017.zip
  unzip -q -o downloads/val2017.zip -d .
fi
if [ ! -s annotations/instances_val2017.json ]; then
  wget -c -P downloads http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q -o -j downloads/annotations_trainval2017.zip \
    annotations/instances_val2017.json -d annotations
fi
n_img="$(ls val2017 | grep -c '\.jpg$')"
[ "$n_img" = 5000 ] || { echo "[setup] val2017 has $n_img images, expected 5000"; exit 1; }
echo "[setup] COCO ok: 5000 images + instances_val2017.json (zips kept in downloads/, delete if short on disk)"

# --- 5. weights: let the libraries fetch them to the paths they read -----
# YOLO .pt lands in the repo root, CLIP ViT-B-32.pt in ./weights/clip/.
[ -s yolov8x-worldv2.pt ] && [ -s weights/clip/ViT-B-32.pt ] || "$PY" -c "
from ultralytics import YOLO
m = YOLO('yolov8x-worldv2.pt'); m.set_classes(['person'])
print('[setup] YOLO-World + CLIP ok')"
"$PY" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('microsoft/Florence-2-base', revision='refs/pr/26')
print('[setup] Florence-2-base @ refs/pr/26 ->', p)"

# --- 6. checks -------------------------------------------------------------
"$PY" phase4/p4_yolo.py --preflight
"$PY" phase4/p4_florence.py --preflight
"$PY" tests/test_offline_logic.py | tail -2
"$PY" tests/test_flow_stubbed.py | tail -2
CUDA_VISIBLE_DEVICES='' "$PY" tests/test_attack_units.py | tail -2

cat <<EOF

[setup] done. Next, on a GPU node:
  conda activate $ENV_NAME && cd $REPO && export HF_HOME=$REPO/weights/hf
  export HF_HUB_OFFLINE=1   # only if the GPU node has no internet
  python phase4/p4_yolo.py --experiment adaptive_bpda --smoke-test --device cuda --gpu 0
  python phase4/p4_florence.py --experiment clean_control --smoke-test --device cuda --gpu 0
EOF
