# Robustifying Zero-Shot Detection and OCR Against Adversarial Attacks

This repository studies adversarial robustness for two open-vocabulary models:
- Florence-2-Base (sequence-to-sequence VLM)
- YOLOv8x-WorldV2 (one-stage detector)

Attacks: FGSM, PGD, Patch

Defenses: solo transforms + detection-level transform ensembles merged by class-aware NMS.

Primary report: `CS24MTECH14020_CVPR_Project_Report.pdf`

## Current Status (Important)

- **Phase 3 v2 at N=1000 is complete for all 9 tracks** (3 attacks × {YOLO-det, Florence-det,
  Florence-OCR}). Use the `*_v2` notebooks/`*_Robust.py` scripts and their `summary.json` for the
  locked paper numbers.
- **Survey-scope N=5000 scripts are ready** (`run_survey_*.py`) — the comprehensive defense survey
  (Tier 1 paper set + Tier 2 completeness/negatives + Tier 3 novel 2025–2026 methods).
- Older Phase 3 outputs without `_v2` are historical and may include contamination from checkpoint
  reuse — do **not** cite them. The survey scripts use a config-keyed `SurveyCheckpoint` that cannot
  reproduce that contamination.

## Team

| Member | Track | Script |
|---|---|---|
| Ankush | Florence-2 OCR | `run_survey_florence_ocr.py` |
| Digvijay | YOLOv8x-worldv2 detection | `run_survey_yolo.py` |
| Lokendra | Florence-2 detection | `run_survey_florence_detection.py` |

Each member runs independently on an RTX GPU (`conda activate vlm_ftune`).

## Repository Guide: Important vs Extra

You asked for clarity on this, so here is the practical split.

### A) Important current files (root-level)

Core docs:
- `CS24MTECH14020_CVPR_Project_Report.pdf` (main paper/report)
- `Presentation.pdf`
- `memory.md` (project state + experiment notes)
- `README.md`

Core code/shared utilities:
- `phase3_common.py`

Active Phase 3 v2 notebooks (detection):
- `FGSM_Phase3_Florence_v2.ipynb`
- `PGD_Phase3_Florence_v2.ipynb`
- `Patch_Phase3_Florence_v2.ipynb`
- `FGSM_Phase3_YOLO_v2.ipynb`
- `PGD_Phase3_YOLO_v2.ipynb`
- `Patch_Phase3_YOLO_v2.ipynb`

Active OCR robustness scripts (Florence):
- `FGSM_Florence2_OCR_Robust.py`
- `PGD_Florence2_OCR_Robust.py`
- `Patch_Florence2_OCR_Robust.py`

Primary result folders (v2):
- `results_phase3_florence_fgsm_v2/`
- `results_phase3_florence_pgd_v2/`
- `results_phase3_florence_patch_v2/`
- `results_phase3_yolo_fgsm_v2/`
- `results_phase3_yolo_pgd_v2/`
- `results_phase3_yolo_patch_v2/`
- `results_fgsm_florence2_ocr_robust/`
- `results_pgd_florence2_ocr_robust/`
- `results_patch_florence2_ocr_robust/`

Data/model assets used by runs:
- `val2017/`
- `annotations/`
- `yolov8x-worldv2.pt`
- `environment.yml`

### B) Extra/archival content (not primary entry points)

- `Scripts_Extra/` (older/variant scripts and notebooks)
- `Logs_Extra/` (saved logs and historical result snapshots)
- `Very_OLD/` and other archival folders

Keep these for traceability, but treat root-level v2 notebooks/scripts as the main execution path.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate vlm_ftune
python -m ipykernel install --user --name=vlm_ftune --display-name "Python (vlm_ftune)"
```

## Dataset Layout

Current notebooks expect COCO validation data at repository root:

```text
./val2017/
./annotations/instances_val2017.json
```

If your local path differs, update variables in the notebook/script config cells.

## Quick Run: Detection (Phase 3 v2)

```bash
conda activate vlm_ftune
jupyter lab
```

Open and run one notebook end-to-end (for example):
- `FGSM_Phase3_YOLO_v2.ipynb`
- `PGD_Phase3_YOLO_v2.ipynb`
- `Patch_Phase3_YOLO_v2.ipynb`

Then run Florence v2 notebooks similarly.

## Quick Run: Florence OCR Robustness Scripts

The OCR scripts default to `dataset/val2017`, so in this repo pass `--image-dir ./val2017`.

FGSM OCR:
```bash
conda activate vlm_ftune
python FGSM_Florence2_OCR_Robust.py \
  --image-dir ./val2017 \
  --num-images 1000 \
  --eps 0.03 \
  --output-dir results_fgsm_florence2_ocr_robust
```

PGD OCR:
```bash
conda activate vlm_ftune
python PGD_Florence2_OCR_Robust.py \
  --image-dir ./val2017 \
  --num-images 1000 \
  --eps 0.03 \
  --steps 10 \
  --alpha 0.003 \
  --output-dir results_pgd_florence2_ocr_robust
```

Patch OCR:
```bash
conda activate vlm_ftune
python Patch_Florence2_OCR_Robust.py \
  --image-dir ./val2017 \
  --num-images 1000 \
  --patch-size 35 \
  --patch-iters 100 \
  --patch-lr 0.02 \
  --output-dir results_patch_florence2_ocr_robust
```

## Survey Scripts (N=5000, comprehensive defense survey)

Three consolidated CLI scripts run **all three attacks in one session** and sweep the full survey
defense bank (Tier 1 paper set + svd, Tier 2 completeness/negatives, Tier 3 novel via `--novel`).
All import `phase3_common.py`; checkpoints are config-keyed and resumable (and gitignored).

```bash
conda activate vlm_ftune

# YOLOv8x-worldv2 detection (Digvijay)
python run_survey_yolo.py \
  --image-dir ./val2017 --ann-file ./annotations/instances_val2017.json \
  --gpu 0 --num-images 5000 --tier survey --attacks fgsm pgd patch

# Florence-2 detection (Lokendra) — note the ./Dataset/ layout these scripts expect
python run_survey_florence_detection.py \
  --image-dir ./Dataset/val2017 --ann-file ./Dataset/annotations/instances_val2017.json \
  --gpu 0 --num-images 5000 --tier survey --attacks fgsm pgd patch

# Florence-2 OCR (Ankush)
python run_survey_florence_ocr.py \
  --image-dir ./val2017 --gpu 0 --num-images 5000 --tier survey --attacks fgsm pgd patch
```

Flags shared by all three: `--tier {tier1,survey}`, `--attacks {fgsm,pgd,patch}...`,
`--checkpoint-every N`, `--no-checkpoint`, `--novel` (attempt Tier-3 methods, skip gracefully if
their deps are missing). Each writes one `summary_{attack}.json` per attack into
`results_survey_{yolo,florence_detection,florence_ocr}/`.

> **OCR metric note:** the OCR track reports **self-consistency** — char-level `SequenceMatcher`
> similarity of the defended OCR text vs the model's *own clean* OCR output (COCO has no OCR ground
> truth). It is prediction-stability recovery, **not** OCR accuracy.

## Repository Size and Git Tracking

COCO per-condition eval JSONs are large and regenerable from `detections.pkl`; only the derived
`summary*.json` and `run.log` are version-controlled. `.gitignore` ignores `results_*/*.json` with a
`!results_*/summary*.json` exception (the `summary*` glob keeps per-attack survey outputs such as
`summary_fgsm.json`), plus survey checkpoints and all `*.pkl`. The 414 previously-tracked dump JSONs
have been untracked (`git rm --cached`; files remain on disk). For a future public release, large
blobs can be purged from history with **BFG Repo Cleaner** (deferred; all team members must re-clone
afterward — see `memory.md`).

## Result Artifacts and Versioning Notes

- `.pkl` outputs are large runtime artifacts and are typically not tracked.
- JSON summaries and comparison outputs are the primary files for analysis/reporting.
- For writing the paper/report, use v2 outputs and cross-check with:
  - `CS24MTECH14020_CVPR_Project_Report.pdf`
  - `memory.md`

## Confirmed v2 Snapshot (N=1000, read from committed `summary.json`)

Detection = COCO mAP@[.5:.95]; OCR = self-consistency similarity.

| Track | clean | attacked | best defense | defended |
|---|---|---|---|---|
| YOLO · FGSM | 0.4765 | 0.2259 | blur_tvm | 0.3448 (+47.4%) |
| YOLO · PGD (10 iters) | 0.4765 | 0.0737 | ens_blur_tvm_combo | 0.4128 (+84.2%) |
| YOLO · Patch | 0.4765 | 0.4537 | ens_jpeg_median_tvm | 0.4599 (+27.5%) |
| Florence · FGSM | 0.3605 | 0.2731 | ens_jpeg_median_gaussian | 0.3281 (+62.9%) |
| Florence · PGD | 0.3605 | 0.2243 | ens_jpeg_median_gaussian | 0.3245 (+73.5%) |
| Florence · Patch | 0.3605 | 0.3268 | ens_blur_tvm_combo | 0.3324 (+16.7%) |

> **Correction:** earlier notes quoted YOLO PGD as `clean 0.4972 → 0.1425 → 0.4801` and a
> "stronger PGD" `0.4765 → 0.0614 → 0.4263`. Those match **no committed v2 result** (they trace to a
> 100-image scratch run) — the reproducible v2 PGD numbers are the table above. Cite only committed
> `summary.json` values.

Interpretation rule used in this repo:
- Prioritize **recovery from attacked mAP** over naive comparison to clean baseline.

## Citation

If you use this repository, cite the project report in:
- `CS24MTECH14020_CVPR_Project_Report.pdf`

And keep the model/data citations used in the report (Florence-2, YOLO-World, COCO, FGSM, PGD, Patch attack references).