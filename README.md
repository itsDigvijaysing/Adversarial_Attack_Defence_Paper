# Robustifying Zero-Shot Detection and OCR Against Adversarial Attacks

This repository studies adversarial robustness for two open-vocabulary models:
- Florence-2-Base (sequence-to-sequence VLM)
- YOLOv8x-WorldV2 (one-stage detector)

Attacks: FGSM, PGD, Patch

Defenses: solo transforms + detection-level transform ensembles merged by class-aware NMS.

Primary report: `CS24MTECH14020_CVPR_Project_Report.pdf`

## Current Status (Important)

- The current authoritative pipeline is **Phase 3 v2**.
- Use `*_v2` notebooks/results for reporting numbers.
- Older Phase 3 outputs without `_v2` are historical and may include contamination from checkpoint reuse.
- Based on current project notes + report alignment:
  - YOLO v2 runs are complete and consistent.
  - Florence v2 detection/OCR runs are the active track.

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

## Result Artifacts and Versioning Notes

- `.pkl` outputs are large runtime artifacts and are typically not tracked.
- JSON summaries and comparison outputs are the primary files for analysis/reporting.
- For writing the paper/report, use v2 outputs and cross-check with:
  - `CS24MTECH14020_CVPR_Project_Report.pdf`
  - `memory.md`

## Confirmed YOLO v2 Snapshot (from project notes)

- FGSM (eps=0.03): clean 0.4765 -> attacked 0.2259, best defended 0.3450
- PGD (eps=0.03, 10 iters): clean 0.4972 -> attacked 0.1425, best defended 0.4801
- Stronger PGD setting: clean 0.4765 -> attacked 0.0614, best defended 0.4263

Interpretation rule used in this repo:
- Prioritize **recovery from attacked mAP** over naive comparison to clean baseline.

## Citation

If you use this repository, cite the project report in:
- `CS24MTECH14020_CVPR_Project_Report.pdf`

And keep the model/data citations used in the report (Florence-2, YOLO-World, COCO, FGSM, PGD, Patch attack references).