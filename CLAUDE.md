# CLAUDE.md — Adversarial Attack/Defence Survey

Guidance for AI assistants working in this repo. See `README.md` (usage) and `memory.md`
(project state, results, citations). Phase-1/2 history is in `memory_phase2_archive.md`.

## What this project is
A survey of input-transform adversarial defenses for two open-vocabulary models —
**Florence-2-base** (VLM: object detection + OCR) and **YOLOv8x-worldv2** (object detector) — on
**COCO val2017**, under **FGSM / PGD / Patch** attacks. Two evaluation tracks: **detection**
(COCO mAP@[.5:.95]) and **OCR** (text self-consistency). Scope explicitly includes **negative
results** and **novel 2025–2026 methods**.

## Environment & run
- `conda activate vlm_ftune` (PyTorch 2.6.0+cu118, transformers 4.51.0, see `environment.yml`).
- Models: `microsoft/Florence-2-base` revision `refs/pr/26` (float16); `yolov8x-worldv2.pt`
  (ultralytics auto-downloads).
- Data (gitignored, fetch separately): COCO `val2017/` + `annotations/instances_val2017.json`.
- Survey runs: `python run_survey_{yolo,florence_detection,florence_ocr}.py --tier survey
  --num-images 5000 --attacks fgsm pgd patch` (see README for full CLI).
- **Do NOT execute training/eval scripts unless explicitly asked** — they need GPU + model + COCO.
  For code changes, verify with `python -m py_compile <file>` (syntax only).

## Layout
- **`phase3_common.py`** — shared core: GPU defenses (`tvm_gpu`, `median_gpu`, `gaussian_gpu`,
  `svd_gpu`, `bilateral_gpu`), CPU defenses (`jpeg_cpu`, `nlm_cpu`, `bit_depth_cpu`,
  `random_resize_pad_cpu`, `dithering_cpu`, `anisotropic_diffusion_cpu`, optional `bm3d_cpu`),
  `apply_all_defenses_gpu` (5 locked solos) / `apply_survey_defenses` (full survey bank),
  `merge_branches_nms`, `assemble_results`, COCO eval, and `SurveyCheckpoint` + `config_signature`.
- **Locked v2 (N=1000, paper results) — DO NOT MODIFY:** `{FGSM,PGD,Patch}_Phase3_Florence_v2.{py,ipynb}`,
  `{FGSM,PGD,Patch}_Phase3_YOLO_v2.ipynb`, `{FGSM,PGD,Patch}_Florence2_OCR_Robust.py`.
- **Survey scripts (N=5000):** `run_survey_{yolo,florence_detection,florence_ocr}.py`.
- **`docs/`** — report + presentation PDFs, `GPU`, `tmux.txt`. **`figures/`** — result images.
- **`archive/`** — read-only history: `Scripts_Extra/`, `Logs_Extra/` (3.2 GB of historical dumps,
  gitignored), `Very_OLD/`. No active code reads from `archive/`.

## Critical conventions & gotchas
- **Never modify the frozen v2 files or `pc.SOLO_DEFENSES`/`pc.ENSEMBLES`.** v2 scripts iterate
  those 5 solos / 3 ensembles and `apply_all_defenses_gpu` produces exactly those keys — adding a
  name (e.g. `svd`) KeyErrors every v2 run. Survey defenses live in `TIER*/SURVEY_*` constants.
- **Florence detection scores are fabricated** (`_compute_score`: geometry, not confidence) because
  Florence `<OD>` emits no scores. Applied identically to all conditions → deltas fair, absolute mAP
  geometry-ranked. YOLO uses real confidences. Disclose this in the paper.
- **OCR metric = self-consistency**, not accuracy: char-level `SequenceMatcher` vs the model's *own
  clean* OCR output (no COCO OCR GT). `clean_baseline` ≈ 1.0 by construction. Never call it
  "word-recovery ratio."
- **Attack pixel spaces differ:** Florence attacks in **normalized** (ImageNet) space, clamp
  [−2.5,2.5]; YOLO in **raw [0,1]**.
- **Checkpointing:** use the config-keyed `SurveyCheckpoint`, never the legacy `DetectionCheckpoint`
  with bare `has()`/skip — that caused the documented v1 cross-run contamination. New survey runs
  derive `defense_names` from the producer's returned keys (bm3d may be absent).
- **Cite only committed `summary.json` numbers.** Older PGD-YOLO figures in past notes were stale
  (no committed file backs them); correct v2 PGD = 0.4765→0.0737→0.4128 (ens_blur_tvm_combo).
- **Dataset paths are inconsistent by track:** YOLO/OCR use `./val2017`; Florence detection uses
  `./Dataset/val2017`. Match the track you're editing.
- **Novel-method citations are verified but mostly NOT evaluated on Florence-2** (only EigenShield
  is). Cite preprints as preprints; see the table in `memory.md`.

## Git
- **Do not commit unless explicitly asked.** Per the user's global preference, **never add
  `Co-Authored-By` or any AI/assistant attribution** to commits/PRs.
- Only `summary*.json`, `run.log`, `.png`, code, and docs are tracked under `results_*/`; dump JSONs,
  `*.pkl`, and survey checkpoints are gitignored.
