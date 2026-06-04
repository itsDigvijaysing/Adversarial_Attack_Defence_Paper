# Project Memory — Adversarial Robustness Survey (Florence-2 + YOLO-World)

> Phase-1/Phase-2 narrative lives in **`memory_phase2_archive.md`** (archived). This file is the
> survey-scope reference. All numbers below are read from committed `summary.json` files.

## Project State
A comprehensive **survey** of input-transform adversarial defenses for two open-vocabulary
models — **Florence-2-base** (VLM: object detection + OCR) and **YOLOv8x-worldv2** (object
detector) — on **COCO val2017**, under three attacks (**FGSM**, **PGD**, **Patch**). Scope has
expanded from the locked paper set to cover *all* defenses tried, **including documented
negative results** (research value), plus **novel 2025–2026 methods** from the literature. The
APSCON IEEE submission used the earlier Phase-1/2 work; the report/presentation PDFs
(`docs/CS24MTECH14020_CVPR_Project_Report.pdf`, `docs/Presentation.pdf`) live in `docs/`.
**Phase-3 v2 at N=1000 is complete for all 9 tracks** (3 attacks × {YOLO-det, Florence-det,
Florence-OCR}); **survey runs at N=5000 are the active work** via the three `run_survey_*.py`
scripts.

## Team
| Member | Track | Script | GPU |
|---|---|---|---|
| Ankush | Florence-2 OCR | `run_survey_florence_ocr.py` | RTX |
| Digvijay | YOLOv8x-worldv2 detection | `run_survey_yolo.py` | RTX |
| Lokendra | Florence-2 detection | `run_survey_florence_detection.py` | RTX |
Each runs independently (`conda activate vlm_ftune`).

## Active Files
- **Shared core:** `phase3_common.py` (GPU defenses, survey defense bank `apply_survey_defenses`,
  tiered registry, `merge_branches_nms`, `assemble_results`, COCO eval, `SurveyCheckpoint`).
- **Survey scripts (N=5000):** `run_survey_yolo.py`, `run_survey_florence_detection.py`,
  `run_survey_florence_ocr.py`.
- **Locked v2 (N=1000, paper main results — DO NOT MODIFY):** `{FGSM,PGD,Patch}_Phase3_Florence_v2.{py,ipynb}`,
  `{FGSM,PGD,Patch}_Phase3_YOLO_v2.ipynb`, `{FGSM,PGD,Patch}_Florence2_OCR_Robust.py`.
- **Output dirs:** `results_phase3_{yolo,florence}_{fgsm,pgd,patch}_v2/`,
  `results_{fgsm,pgd,patch}_florence2_ocr_robust/`, and the new
  `results_survey_{yolo,florence_detection,florence_ocr}/`.
- **Archival (`archive/`):** `archive/Scripts_Extra/` (variant/older scripts),
  `archive/Logs_Extra/` (historical logs, 3.2 GB, gitignored), `archive/Very_OLD/` (Phase-1).
- **`docs/`** (PDFs, `GPU`, `tmux.txt`), **`figures/`** (`Final_Results_Images/`, `Result_Images/`).

## Defense Taxonomy (Survey)
Status: **active** (in paper), **dropped** (documented negative), **new** (added for survey,
not yet swept at N=5000). Recovery% = Florence-2 detection FGSM, N=1000, where measured.

### Tier 1 — paper main set (locked v2 = 5 solos + 3 ensembles; +svd & 2 ensembles for survey)
| Defense | Reference | Status | Florence-FGSM rec% | Notes |
|---|---|---|---|---|
| jpeg (q=75) | Dziugaite 2016 | active | +53.5% | strong on VLM |
| median (3×3) | Xu et al. NDSS 2018 | active | +52.9% | strong |
| tvm (w=0.05) | Guo et al. ICLR 2018 | active | **−11.4%** | *hurts* single-step FGSM (semantic-grad alignment) |
| gaussian (σ=1.0) | Xu et al. NDSS 2018 | active | +13.4% | |
| blur_tvm | (composite) | active | +1.6% | tvm drags it on FGSM |
| svd (90%) | spectral truncation | active (survey) | n/a (v2 omits) | Tier-1 survey-only; not in locked v2 |
| ens_blur_tvm_combo / ens_jpeg_median_gaussian / ens_jpeg_median_tvm | NMS merge | active | +60.8 / +62.9 / +60.6% | ensembles dominate on Florence |
| ens_4way, ens_jpeg_tvm_svd | NMS merge | new | — | survey-only |

### Tier 2 — survey completeness (incl. negatives)
| Defense | Reference | Status | Notes |
|---|---|---|---|
| nlm | Buades et al. CVPR 2005 | dropped | negative recovery in Phase-2 (failed despite CAAD 2018 win); h fixed to 0.8·σ |
| bit_depth (4-bit) | Xu et al. NDSS 2018 | dropped | negative strict gain in Phase-2 |
| random_resize+pad | Xie et al. ICLR 2018 | dropped | worst Phase-2 performer; now seeded for reproducibility |
| bilateral | Tomasi & Manduchi ICCV 1998 | new | **true** joint spatial+range filter (fixed from a mislabeled blur-blend) |
| dithering | ordered Bayer (fast F-S stand-in) | new | vectorized; serial Floyd-Steinberg was ~30–40h infeasible |
| anisotropic | Perona & Malik PAMI 1990 | new | kappa fixed to 0.05 on [0,1] data (kappa=50 → plain blur) |
| bm3d | Dabov et al. TIP 2007 | new (optional) | needs `pip install bm3d`; **dropped with a warning if absent** (never a mislabeled row) |

### Tier 3 — novel 2025–2026 (see next section)
Attempted via `--novel`; dependency-guarded, skip-with-warning if unavailable.

## Attack Specifications (locked)
- **FGSM** — eps=0.03, 1 step.
- **PGD** — eps=0.03, 10 iters, alpha=eps/4, random start, project to eps-ball each step.
- **Patch** — 35×35 center patch, 100 Adam steps, lr=0.02.
- **Pixel space:** Florence attacks in **normalized** (ImageNet mean/std) space, clamp [−2.5,2.5];
  YOLO attacks in **raw [0,1]**.

## Current Results (N=1000 v2) — read from committed summary.json
| Track | clean | attacked | best defense | defended | recovery |
|---|---|---|---|---|---|
| YOLO · FGSM | 0.4765 | 0.2259 | blur_tvm | 0.3448 | +47.4% |
| YOLO · PGD | 0.4765 | **0.0737** | **ens_blur_tvm_combo** | **0.4128** | **+84.2%** |
| YOLO · Patch | 0.4765 | 0.4537 | ens_jpeg_median_tvm | 0.4599 | +27.5% |
| Florence · FGSM | 0.3605 | 0.2731 | ens_jpeg_median_gaussian | 0.3281 | +62.9% |
| Florence · PGD | 0.3605 | 0.2243 | ens_jpeg_median_gaussian | 0.3245 | +73.5% |
| Florence · Patch | 0.3605 | 0.3268 | ens_blur_tvm_combo | 0.3324 | +16.7% |
| OCR · FGSM | 1.0000 | 0.3287 | tvm | 0.5615 | +34.7% |
| OCR · PGD | 1.0000 | 0.3139 | tvm | 0.6384 | +47.3% |
| OCR · Patch | 1.0000 | 0.3675 | median | 0.5566 | +29.9% |

- **Detection** = COCO mAP@[.5:.95] (pycocotools). **Florence detection mAP uses fabricated
  geometric confidence scores** (`_compute_score`: area-ratio + centeredness) because Florence
  `<OD>` emits no scores — applied identically to all conditions, so *deltas* are fair but the
  *absolute* Florence mAP is geometry-ranked (must be disclosed in the paper). YOLO uses real
  confidences.
- **OCR** = **self-consistency**: char-level `difflib.SequenceMatcher` ratio of defended OCR text
  vs the model's **own clean-image OCR output** (no COCO OCR ground truth exists). `clean_baseline`
  is 1.0 by construction. This is prediction-stability recovery, **NOT OCR accuracy** — never label
  it "word-recovery ratio".
- ⚠️ **Stale-number correction:** the old README/memory PGD-YOLO figures (0.4972/0.1425/0.4801 and
  0.4765/0.0614/0.4263) match **no committed v2 file** (traceable only to a 100-image
  `archive/Logs_Extra/` run). The reproducible v2 PGD result is the bold row above. Cite only these committed numbers.

## Novel 2025–2026 Methods (verified citations — fix before citing in the paper)
| Method | Real citation | Eval target | Caveat |
|---|---|---|---|
| **EigenShield** | arXiv:2502.14976 → **AAAI-26 (2026)**, causal-subspace RMT filtering | **Florence-2-large ✓** | cite as AAAI **2026** not 2025; preprint vs published titles differ |
| **XAIAD-YOLO** | Future Generation Computer Systems (Elsevier) **2026**, DOI 10.1016/j.future.2025.108356 | **YOLO only** | not a VLM/Florence method |
| **PAD** | **CVPR 2024**, arXiv:2404.16452, SAM-based patch defense | object detectors | not evaluated on Florence-2 |
| **SIGN** | arXiv:2605.27927 (**2026 preprint**); title "Structure-Guided Visual Perturbation Neutralization for LVLMs" | LVLMs | preprint (no venue); 0.5%-pixel claim ✓; acronym ≠ "structure-guided perturbation neutralization" |
| **VALD** | arXiv:2602.19570 (**2026 preprint**), multi-stage LVLM defense | LLaVA/MiniGPT/Qwen | preprint; **not** Florence-2 |
| **DisPatch** | arXiv:2509.04597 (**2025 preprint**), diffusion patch defense for OD | object detectors | cite as preprint |
| **Saliuitl** | **CVPR 2025**, ensemble-salience patch recovery | CNNs | not a VLM method |
**All seven exist** (web-verified). **Only EigenShield was actually tested on Florence-2** — do not
imply Florence-2 evaluation for the others. Cite preprints as preprints.

## Key Lessons (carried forward)
- **v1 contamination root cause:** `DetectionCheckpoint` + `if ckpt.has(img_id): return` over a
  single shared pickle silently reused stale entries across configs → FGSM negative recovery, PGD
  ensembles > clean, num_images misreport. **v2 fix = no cache.** The survey scripts use
  `SurveyCheckpoint`, which is **config-signature-keyed** (discards a resumed cache whose
  attack/defense/param signature differs), so it cannot reproduce that contamination.
- **FGSM vs PGD recovery asymmetry is expected physics, not a bug:** iterative PGD → high-freq
  noise that smoothing defenses wipe (high recovery); single-step FGSM aligns with semantic
  gradients, so smoothing damages signal (lower recovery — see tvm −11.4% on Florence FGSM).
- **No batching `generate(num_beams=5)`** — per-image (optionally dual-GPU threading); batching
  thrashes VRAM with little wall-clock gain.
- **Net-gain interpretation:** judge defenses by recovery vs the *attacked* floor, not the clean
  baseline.

## Repository Size Note
COCO per-condition eval JSONs are large and regenerable from `detections.pkl`; only `summary*.json`
+ `run.log` are tracked. `.gitignore` now: ignores `results_*/*.json` with `!results_*/summary*.json`
(the `summary*` glob keeps per-attack survey summaries like `summary_fgsm.json`), ignores survey
checkpoints (`results_survey_*/checkpoint_*.pkl`, `state_*.json`) and `*.pkl`. 414 previously-tracked
dump JSONs were untracked (`git rm --cached`, files kept on disk). **Optional, deferred:** after the
paper/final sync, purge large blobs from history with **BFG Repo Cleaner**
(`java -jar bfg.jar --delete-files '*.json' --no-blob-protection .` — ⚠️ this also deletes
`summary.json` from history; scope it to dump names if you want summaries preserved — then
`git reflog expire --expire=now --all && git gc --prune=now --aggressive`; **all members must
re-clone** afterward).

## Next Steps
1. Run the 3 survey scripts at **N=5000** (`--tier survey`), one per team member/GPU.
2. Collect all `summary_{attack}.json` into a master results table (Tier-1 vs Tier-2 vs Tier-3).
3. Write the survey sections, **including the negative results** (nlm/bit_depth/random_resize) and
   the FGSM-vs-PGD asymmetry.
4. Add Tier-3 novel-method comparisons (`--novel`), citing each correctly (preprints as preprints).
5. BFG history cleanup before any public GitHub publish.
