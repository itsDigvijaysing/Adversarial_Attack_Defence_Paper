# Project Memory — Adversarial Robustness Survey (Florence-2 + YOLO-World)

> Phase-1/Phase-2 narrative lives in **`memory_phase2_archive.md`** (archived). This file is the
> survey-scope reference. All numbers below are read from committed `summary.json` files unless
> marked off-repo. **Last full update: 2026-10-26** (Phase-4 code ready, CPU rescore, novelty check;
> 22 Oct: submission history, PDF audit, Phase-4 status,
> revival plan).

## Project State (2026-10-22)
A **benchmark of input-transform adversarial defenses** for two open-vocabulary models,
**Florence-2-base** (VLM: object detection + OCR) and **YOLOv8x-worldv2** (object detector), on
**COCO val2017**, under three attacks (**FGSM**, **PGD**, **Patch**), including documented negative
results. Where it stands:
1. **Phase 1/2:** published as the IEEE APSCON 2026 paper "Evaluating and Mitigating Adversarial
   Vulnerabilities in Vision Foundation Models" (archival). The report/presentation PDFs are in
   `docs/`.
2. **Phase 3:** v2 at N=1000 complete for all 9 tracks (3 attacks × {YOLO-det, Florence-det,
   Florence-OCR}). Survey at N=5000 complete for all 3 tracks; only the Florence-det N=5000
   summaries are committed (YOLO and OCR live on teammates' machines).
3. **Submitted and rejected:** ECCV 2026 workshop EmergingAD (archival), #8, submitted 20 Jul 2026,
   rejected 8 Aug 2026. Not ICLR. See "Submission History & Reviews".
4. **Phase 4 (since 17 Aug):** `phase4/` adds the experiments the reviews asked for
   (`clean_control`, `adaptive_bpda`). Only smoke tests have run on GPU; the CPU rescore of the
   N=1000 dumps is done. See "Phase-4 Status" and "Rescore Findings".
5. **Goal now:** revive it as an evaluation paper. Target TMLR (~Mar-Apr 2027), fallback the WACV
   2028 Evaluations & Datasets track. See "Revival Plan & Venues".

## Team
| Member | Track | Phase-4 command | GPU |
|---|---|---|---|
| Ankush | Florence-2 OCR | `python phase4/p4_florence.py --tracks florence_ocr` | RTX |
| Digvijay | YOLOv8x-worldv2 detection | `python phase4/p4_yolo.py` | RTX |
| Lokendra | Florence-2 detection | `python phase4/p4_florence.py --tracks florence_det` | RTX |

Each runs independently (`conda activate vlm_ftune`). Digvijay is first author on the
ECCV-W submission.

## Active Files
- **Shared core (stays at the repo root):** `phase3_common.py` (GPU defenses, survey defense bank
  `apply_survey_defenses`, tiered registry, `merge_branches_nms`, `assemble_results`, COCO eval,
  `SurveyCheckpoint`).
- **Phase 4 (active):** `phase4/p4_core.py`, `p4_yolo.py`, `p4_florence.py`, `p4_logging.py`;
  tests in `tests/`. Outputs go to `results/phase4/<experiment>_<model>/`.
- **Phase 3 (archived, read-only):** `archive/Phase3/` holds the locked v2 scripts/notebooks
  (`{FGSM,PGD,Patch}_Phase3_Florence_v2.py`, `{FGSM,PGD,Patch}_Phase3_YOLO_v2.ipynb`,
  `{FGSM,PGD,Patch}_Florence2_OCR_Robust.py`) and the N=5000 survey scripts
  (`run_survey_{yolo,florence_detection,florence_ocr}.py`).
- **Output dirs:** `results/results_phase3_{yolo,florence}_{fgsm,pgd,patch}_v2/`,
  `results/results_{fgsm,pgd,patch}_florence2_ocr_robust/`,
  `results/results_survey_florence_detection_{fgsm,pgd,patch}/` (N=5000). The suffix-less
  `results/results_survey_florence_detection/` is a 3-image smoke run, not a result.
- **Submitted paper PDF (not in the repo):** `/home/king/Downloads/IMP Files/CV_Research_ECCV.pdf`
  (15 pp, built 18 Jul 2026). The LaTeX source is on Overleaf.
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
| svd (90%) | spectral truncation | active (survey) | n/a (v2 omits) | Tier-1 survey-only; not in locked v2. **Keeps `int(0.9·rank)` singular values by count, not 90% of the energy** (`svd_gpu`), so it is close to identity; the paper's "90% energy" is wrong |
| ens_blur_tvm_combo / ens_jpeg_median_gaussian / ens_jpeg_median_tvm | NMS merge | active | +60.8 / +62.9 / +60.6% | ensembles dominate on Florence |
| ens_4way, ens_jpeg_tvm_svd | NMS merge | new | — | survey-only |

### Tier 2 — survey completeness (incl. negatives)
| Defense | Reference | Status | Notes |
|---|---|---|---|
| nlm | Buades et al. CVPR 2005 | dropped | negative recovery in Phase-2; h fixed to 0.8·σ (the paper says h=6: wrong). The CAAD 2018 winner the paper cites as [22] used in-network feature denoising with adversarial training, not pixel NLM, so this result does not test that ranking |
| bit_depth (4-bit) | Xu et al. NDSS 2018 | dropped | negative strict gain in Phase-2 |
| random_resize+pad | Xie et al. ICLR 2018 | dropped | worst Phase-2 performer; now seeded for reproducibility. **Detection result is an evaluation artefact:** `random_resize_pad_cpu` shrinks and pastes the image at an offset but boxes are never mapped back, so they are scored against misaligned GT. Fix the inverse mapping or drop it from detection tables |
| bilateral | Tomasi & Manduchi ICCV 1998 | new | **true** joint spatial+range filter (fixed from a mislabeled blur-blend) |
| dithering | ordered Bayer (fast F-S stand-in) | new | vectorized; serial Floyd-Steinberg was ~30–40h infeasible |
| anisotropic | Perona & Malik PAMI 1990 | new | kappa fixed to 0.05 on [0,1] data (kappa=50 → plain blur) |
| bm3d | Dabov et al. TIP 2007 | new (optional) | needs `pip install bm3d`; **dropped with a warning if absent** (never a mislabeled row) |

### Tier 3 — novel 2025–2026 (see next section)
Attempted via `--novel`; dependency-guarded, skip-with-warning if unavailable.

## Attack Specifications (locked)
- **FGSM** — eps=0.03, 1 step.
- **PGD** — eps=0.03, 10 iters, random start, project to eps-ball each step.
  **alpha is NOT uniform:** eps/4 = 0.0075 on every detection track
  (`PGD_Phase3_YOLO_v2.ipynb`, `PGD_Phase3_Florence_v2.py:82`,
  `run_survey_florence_detection.py:167`, `run_survey_yolo.py:161`) **and** in the survey OCR
  script (`run_survey_florence_ocr.py:566`), but **eps/10 = 0.003** in the locked OCR script
  (`PGD_Florence2_OCR_Robust.py:642`), which is the value that produced the committed N=1000 OCR
  PGD row (`results/results_pgd_florence2_ocr_robust/summary.json` → `"alpha": 0.003`). Do not
  write "alpha = eps/4" as a blanket statement, and do not compare the N=1000 OCR PGD number with
  a future N=5000 OCR PGD number without noting the step-size change.
- **Patch** — 35×35 center patch, 100 Adam steps, lr=0.02. Detection scripts use
  `center_patch_coords` at N=1000 and N=5000; OCR used a random location only at N=1000
  (`patch_random_location: true`) and the centre at N=5000 (`random_location=False`). It is a
  per-image **digital** patch (no EOT, no printability term). The submitted paper wrongly says
  "random location per sample" and "physically realizable".
- **Pixel space:** Florence attacks in **normalized** (ImageNet mean/std) space, clamp [−2.5,2.5];
  YOLO attacks in **raw [0,1]**.

### Attack objective — DIFFERENT for the two models (fix this in the paper)
"Token-level cross-entropy against the model's own clean prediction" is true of **Florence-2 only**.
YOLO emits no tokens and has no such loss.
- **Florence-2 (detection `<OD>` and OCR):** self-labeled token-level cross-entropy. The clean image
  is decoded first (`model.generate`, 5 beams, greedy/deterministic) and the resulting id sequence
  becomes `labels`; the attack then ascends the HF seq2seq cross-entropy of the *perturbed* image
  against those frozen self-labels — `out.loss.backward()` for FGSM/PGD,
  `(-out.loss).backward()` + Adam for Patch (Adam minimizes, so the negation still maximizes CE).
  Mean reduction over target tokens. Sources: `FGSM_Phase3_Florence_v2.py:287-298`,
  `PGD_Phase3_Florence_v2.py:289-306`, `Patch_Phase3_Florence_v2.py:294-325`,
  `FGSM_Florence2_OCR_Robust.py:193-209`, `run_survey_florence_detection.py:390-401`.
- **YOLOv8x-worldv2:** confidence suppression on the raw detect head. `pred = model.model(x)[0]` is
  `[1, 4+80, 8400]`; `cls_scores = pred[:, 4:, :]` are **post-sigmoid class probabilities**, and the
  objective is `-cls_scores.max(dim=1)[0].sum()` — a max over the 80 classes per anchor, summed over
  all 8400 anchors. No ground truth, no self-labels, no box/DFL term (`pred[:, :4, :]` is excluded),
  and YOLOv8 is anchor-free so there is **no objectness branch** to include. Sources:
  `run_survey_yolo.py:243-247, 266-270, 303-307`; identical in `{FGSM,PGD,Patch}_Phase3_YOLO_v2.ipynb`
  (cell 14).

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

## Headline Results (N=5000, the numbers the submitted paper uses)
| Track | clean | FGSM | PGD | Patch | Source |
|---|---|---|---|---|---|
| Florence-2 det (mAP) | 0.3300 | 0.2488 | 0.2005 | 0.2958 | committed `results/results_survey_florence_detection_{fgsm,pgd,patch}/summary_*.json` |
| YOLO det (mAP) | 0.4519 | 0.2051 | 0.0619 | 0.4257 | **off-repo** (paper Table 3; Digvijay's machine) |
| Florence-2 OCR (self-consistency) | 1.000 | 0.346 | 0.294 | 0.383 | **off-repo** (paper Table 5; Ankush's machine) |

- The N=5000 PGD alpha is eps/4 on all three tracks; never mix in the N=1000 OCR PGD row (alpha 0.003).
- Paper Table 4 (Florence det, 114 cells) matches the committed files exactly (audit 2026-10-22).
- Paper Table 3 shows Random Resize at 0.1311 under both FGSM and PGD, identical to 4 decimals:
  check the YOLO file for a copy error when it is imported.
- The committed N=1000 numbers differ from these (e.g. YOLO clean 0.4765 vs 0.4519): different
  image subsets, not an error. Never mix the two in one table.

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
- **FGSM vs PGD recovery asymmetry is real but its explanation is a HYPOTHESIS, not a fact.**
  The working story: iterative PGD makes high-frequency noise that smoothing removes, while
  single-step FGSM aligns with semantic gradients. Reviewer 8jGZ flagged it as unvalidated. A rival
  explanation is never addressed: iterative attacks overfit the source model and transfer worse
  than single-step ones (Kurakin et al., ICLR 2017), and the oblivious setting is a transfer to
  model plus transform. Recovery % also divides by a much larger PGD denominator. Either measure
  perturbation spectra or present it as a hypothesis. The same applies to the TVM explanation
  ("removes texture the decoder relies on"): TVM gives +31.3% on Florence PGD, which the story
  does not explain.
- **No batching `generate(num_beams=5)`** — per-image (optionally dual-GPU threading); batching
  thrashes VRAM with little wall-clock gain.
- **Net-gain interpretation:** judge defenses by recovery vs the *attacked* floor, not the clean
  baseline.

## Submission History & Reviews
| Date | Venue | Outcome |
|---|---|---|
| (2026) | IEEE APSCON 2026, "Evaluating and Mitigating Adversarial Vulnerabilities in Vision Foundation Models" (Phase 1/2, Florence-2 FGSM/PGD defenses) | Published, archival. Not CORE-ranked. ~40% content overlap with the ECCV-W version (team estimate) |
| 20 Jul 2026 | ECCV 2026 workshop **EmergingAD**, archival track, #8, "Benchmarking Input-Transform Defenses for Zero-Shot Detection and OCR under Oblivious Adversarial Attacks", topic "Failure Modes, Robustness & Deployment Limitations" | **Rejected 8 Aug 2026** by the Program Chairs |

**Reviews** (summarised; the exact text is not on disk):
- **HFW9, 6/10, confidence 2 (low).** Positive: relevant topic; training-free wrapper defenses are
  pragmatic. Objection: venue fit (digital patches and classical filters may not reflect real AV
  threats; COCO instead of driving data).
- **8jGZ, 4/10, confidence 3.** Positive: important problem; full-COCO evaluation; negative
  results valuable; threat model clearly scoped. Objections: abrupt intro, no research questions,
  reads as loosely connected observations; one run per condition; clean-image performance per
  defense missing; FGSM-vs-PGD and TVM explanations unvalidated; "strictly stronger" imprecise;
  oblivious attacks only; OCR metric measures consistency, not accuracy; YOLO attack loss unclear.

**Diagnosis (2026-10-22):** science first, writing second, venue third. The paper recommends
EnsJMG as a deployment default while its own Sec. 5 expects adaptive attacks to cut recovery
"substantially"; it reports no clean cost and a single run. Venue fit only capped the friendly
score. A rejected workshop paper is not a publication, so it can be resubmitted; the APSCON paper
is the real dual-publication constraint.

## Audit of the Submitted PDF (2026-10-22, checked against code and files)
Must-fix before any resubmission. Line numbers are the PDF margin numbers.
| # | Defect | Evidence |
|---|---|---|
| 1 | **Ref [18] is wrong:** cites arXiv 2403.09761 ("Hydrodynamics of Markets", a finance paper). The real paper is Zhao et al., "One Object, Multiple Lies", ICCV 2025, arXiv 2507.07709 | arXiv API, both IDs checked |
| 2 | **Headline contradicted by the paper's own tables:** "ensembles are the only defenses positive in all nine settings", but Median is positive in all 9 (YOLO +28.3/+74.5/+16.1%; Florence +52.0/+67.7/+21.5%; OCR 0.452/0.546/0.581 vs attacked 0.346/0.294/0.383) | paper Tables 3-5 |
| 3 | YOLO attack loss described as token cross-entropy (l.146); the code is confidence suppression | `run_survey_yolo.py:246-247` |
| 4 | Patch described as random-location and "physically realizable" (l.016, 045, 091, 165, 214); it is a fixed-centre digital patch | `center_patch_coords` in the N=5000 detection scripts |
| 5 | Florence geometric pseudo-scores never disclosed; l.187 "keeps the highest-confidence detection" is false for Florence | `_compute_score` |
| 6 | Random Resize detection numbers are a box-misalignment artefact | `phase3_common.py:254-272` |
| 7 | "SVD (90% energy)" is 90% of rank by count; "NLM (h=6)" is h=0.8σ in code | `phase3_common.py:131-138, 234-251` |
| 8 | OCR ensembles called NMS merges; they are a consensus vote | `run_survey_florence_ocr.py:181-207` |
| 9 | Attack pipeline omitted: processor resize / letterbox, uint8 truncation, bicubic resample back to native size | `run_survey_yolo.py:186-188, 230-232` |
| 10 | Overclaims: "spectral mechanism" (no spectral measurement), "strictly stronger" (PGD with random start is not guaranteed stronger) | l.080-081, l.162-163 |
| 11 | Internal slips: "70-entry lookup table" (89 keys), "Ens4W within a point of EnsJMG" (3.1-point gap), "JPEG and Median are the only solos that help on all three" (Bilateral, Anisotropic also do), EnsJMG "leads both Patch-OD settings" (EnsJBT, EnsBMT beat it on Florence Patch) | paper tables |
| 12 | APSCON is not cited despite the overlap | reference list (23 entries) |

**Checked and NOT a defect:** `FLORENCE_TO_COCO` (89 keys, 30 targets) is only a synonym map;
`_map_label` passes exact COCO names straight through, so all 80 classes are reachable. The
eps-space difference (l.155-160) and the OCR "consistency" wording (l.200-204) are already in
the paper.

**Weak spots a main-venue reviewer will still hit:** the OCR metric on COCO has no text-image
filter (`SequenceMatcher('', '')` = 1.0); "open-vocabulary" is untested with a fixed 80-class
prompt; per-model eps budgets differ ~4.4× (confounds cross-model comparisons); no adaptive
attack on Florence-2.

## Phase-4 Status (2026-10-26)
- **Tests (26 Oct, all CPU, no weights, nothing downloaded):** `tests/test_offline_logic.py` 37/37,
  `tests/test_flow_stubbed.py` 20/20, `tests/test_rescore.py` 20/20 (vlm_ftune; 17 + 3 skipped
  under system python), `tests/test_attack_units.py` 14/14 (CPU torch, fake detector).
- **Ready to run:** `clean_control` (YOLO; Florence det + OCR) and `adaptive_bpda` (YOLO). Adaptive
  flags: `--target` (EnsJMG default, Ens4W, EnsJMT, EnsBTC, solos; members from
  `pc.TIER1_ENSEMBLES`), `--eot-loss {max,mean}` (max default), `--restarts R`,
  `--median-surrogate {true,identity}` (true default), `--quantize {truncate,round}`,
  `--attack-space {letterbox,native}`. Summaries are named by target, seed and every non-default
  flag (`STEM_FLAG_DEFAULTS`).
- **Fixed since 22 Oct:**
  - union-aware loss (max over branches per anchor), matching the NMS-union merge;
  - seeded PGD init (SHA-256 per seed, image and restart), shared by the oblivious and adaptive
    attacks;
  - `--attack-space native` clips the uint8 output to clean +/- 7/255; the letterbox default keeps
    Phase 3's truncation and the `.md` discloses it;
  - `select_images` raises on too few images in both entry points (only `--smoke-test` warns);
    the smoke banner prints the real n and device;
  - `p4_rescore.py` Pool deadlock (`maxtasksperchild=1` inside the `run_log` tee) removed;
  - after the 26 Oct cloud review: `--preflight` counts images against `--n` (it said READY for
    the 20-image dir); the BPDA surrogate choice is one table (`SURROGATES` + `surrogate_spec`)
    instead of four hand-kept copies. The review's "test_attack_units.py is missing" was the file
    being untracked: it, `test_rescore.py` and `p4_rescore.py` must be `git add`ed with any commit.
- **Still open in the attack:** 10 iterations with no convergence check yet (a GPU run);
  letterbox vs native resolution stays the default (the `native` ablation exists).
- **Blockers for real runs:** only **20** val2017 images on disk; the download
  (`http://images.cocodataset.org/zips/val2017.zip`, ~780 MB) is the user's call and was not done
  (user, 24 Oct). No GPU runs for now; the plan is an A6000 if one becomes available.
- **Smoke runs (old, not citable):** ran on `cuda:0` with 2 PGD iterations (1.61 s/img understates
  10 iterations). Adaptive n=2: EnsJMG 0.4966 vs oblivious, 0.3008 vs adaptive; report each
  system's worst case (defended 0.3008 vs undefended 0.2366), not the printed "-0.0796".
- **GPU cost estimate:** clean_control both models + adaptive YOLO (3 seeds, vs EnsJMG, Median,
  Ens4W) + a 50-iteration check: ~15-27 h on the RTX 3060, ~5-7 h on one A6000. Adaptive
  Florence-2 needs ~300-500 new lines plus ~5-10 GPU h.
- **Revision kit:** `docs/paper_revision.md` (742 lines): reviews in brief, errata E1-E22 and
  N1-N11 with PDF lines and LaTeX replacements, method rewrites, RQ1-RQ4 outline, two abstract
  drafts with [X]/[Y] placeholders, verified BibTeX (incl. the ref [18] and [22] fixes), claims to
  soften, a 14-item checklist.

## Rescore Findings (N=1000 Phase-3 v2 dumps, CPU, 24 Oct)
Source: `results/phase4/rescore_phase3_v2/summary_rescore_phase3_v2_latest.{json,md}` (run
`20261024-063133`). Reproduces the committed summary.json values 168/168 (max diff 0). Paired
image-level bootstrap, B=1000, seed 42, 95% CIs. Not the paper's N=5000 set.
- **Clean cost (mAP):**
  - YOLO: every defense costs accuracy, CI excludes 0. Solos -0.015 to -0.029, Blur+TVM -0.062,
    ensembles -0.009 to -0.012 (EnsJMG -0.0109 [-0.0175, -0.0023]).
  - Florence: TVM -0.0715, Gaussian -0.0691, Blur+TVM -0.0954 (large). JPEG, Median and all
    three ensembles: CI spans 0.
- **Patch:** no defense has a recovery CI above 0 in either model. TVM and Blur+TVM (both models),
  JPEG (YOLO) and Gaussian (Florence) make it significantly worse. The patch rows support no
  positive claim.
- **Headline check (6 detection settings):** ensembles are positive on the point estimate in 6/6
  but have a CI above 0 in 4/6, the same as JPEG and Median. Median's point estimate is positive
  in 5/6 (the paper's N=5000 tables: 9/9).
- **Best ensemble minus Median:**
  - YOLO: FGSM +0.0435, PGD +0.0489 [+0.0386, +0.0587], Patch +0.0103 [+0.0017, +0.0161].
  - Florence: FGSM +0.0087 (bias-corrected CI spans 0), PGD +0.0047 [-0.0008, +0.0150], Patch
    +0.0036 (spans 0).
  - So "ensembles beat Median" holds for YOLO only.
- **Score-free F1 (IoU 0.5):**
  - YOLO: read the conf >= 0.25 table, where ensembles roughly match the solos. At the dump
    threshold (0.001) ensembles look worse by construction, because merging three branches adds
    low-confidence boxes.
  - Florence: F1 follows mAP. JPEG, Median and the ensembles recover; TVM, Gaussian and Blur+TVM
    cost 0.07-0.10 F1 on clean images.
- **Caveats:**
  - One attack run per condition, so the CIs cover image sampling only.
  - Single-mAP bootstrap means drift up ~1 SE (AP nonlinearity), so cite the paired-difference
    CIs.
  - Two older timestamped summaries (`_20261024-011847`, `_20261024-012907`) predate the final
    run; cite `_latest`.

## Novelty Check (24 Oct, literature search + refuter agents)
- **Ranking:** none was refuted, and all are narrow.
  1. **Primary:** a union-aware worst-case evaluation of NMS-merged transform ensembles
     (`--eot-loss max`). The closest prior is AdROD, arXiv 2608.16031, which uses mean-EOT only.
     Present it as an application of known worst-case ensemble attacks, not as a new attack.
  2. **Secondary:** a score-rule audit of the Florence pseudo-scores, on CPU (geo vs random vs
     inverse vs constant). It shows how much the ranking depends on `_compute_score`. Not written
     yet.
  3. **Weaker:** a null-view control (fold it into RQ3 only), cross-vocabulary transfer and
     Florence cross-task transfer. Drop the two-model committee.
- **Working title:** "Do Transform Ensembles Survive an Attacker Who Knows the Merge? A Worst-Case
  Re-evaluation of Input-Transform Defenses on YOLO-World and Florence-2".
- **Scoop risks to re-check before submitting:**
  - arXiv 2610.29196 (21 Oct 2026): YOLO-World under patches, no defenses.
  - arXiv 2608.02137.
  - arXiv 2608.16031 (AdROD).
- **Framing rules:**
  - Drop "open-vocabulary" from the novelty claims: only a fixed 80-class prompt is tested.
  - Cite preprints as preprints.
  - The paper is an evaluation, not a survey (TMLR stopped taking surveys on 1 Sep 2026).

## Prior Work to Position Against (searched 2026-10-22)
No paper benchmarks input-transform defenses on YOLO-World or Florence-2 under FGSM/PGD/patch,
and none applies BPDA/EOT to input transforms on open-vocabulary detectors. The gap is narrow:
closed-set and classification work already shows such defenses collapse under adaptive attack.
| Work | Why it matters |
|---|---|
| SAC, CVPR 2022 (paper's ref [10], arXiv 2112.04532) | BPDA sharply reduces JPEG / smoothing defenses for object detectors: direct prior evidence against the paper's recommendation |
| Zheng et al., ICCV 2025, arXiv 2508.00649 | 11 patch defenses on closed-set detectors substantially bypassed by adaptive attacks; no open-vocab models |
| Athalye et al. ICML 2018; Tramèr et al. NeurIPS 2020; Croce et al. ICML 2022 | adaptive-attack methodology; Croce 2022 is missing from the paper |
| Chhipa et al., ECCVW 2024, arXiv 2405.14874 | OWL-ViT / YOLO-World / Grounding DINO under distribution shift, not PGD; no defenses |
| Zhao et al., ICCV 2025 (CrossVLAD), arXiv 2507.07709; arXiv 2608.02137 (Aug 2026) | attacks on Florence-2 and unified VLMs, no defenses |
| Winter et al., arXiv 2602.16494 | unified detector attack benchmark; no open-vocab, no input transforms |
| Lee & Kim, ICCV 2023; AutoAttack, ICML 2020 | defense-overestimation precedents |
| Xu et al., NDSS 2018 (feature squeezing); Kurakin et al., ICLR 2017 | missing citations (squeezing is used uncited; Kurakin is the rival FGSM/PGD explanation) |
| PAD, CVPR 2024 (code: github.com/Lihua-Jing/PAD) | the only runnable modern patch-defense baseline found for both models |

## Revival Plan & Venues (2026-10-22)
- **Framing:** an evaluation paper built on four research questions: RQ1 clean cost per defense;
  RQ2 dependence on model / task / attack, with paired-bootstrap CIs; RQ3 output-level vs
  pixel-level combination at matched compute; RQ4 the adaptivity gap (oblivious vs BPDA+EOT,
  worst case per system).
- **Deciding experiment:** adaptive BPDA+EOT (max-over-branches loss) on YOLO-World, n=1000, 3
  seeds, vs EnsJMG and Median, plus clean_control on both models. If the defenses hold, write a
  positive paper; if they collapse, write "oblivious evaluation overstates input-transform
  robustness on open-vocabulary detectors by X mAP". Both outcomes can be submitted.
- **Venues:**
  - **TMLR** (primary). Rolling, no travel, judges claims not novelty, decision target ~9 weeks.
    It forbids reuse of text, figures or results from archival papers, which makes APSCON the
    risk. Surveys are no longer accepted (since 1 Sep 2026), so do not call this paper a survey.
  - **WACV 2028 Evaluations & Datasets** (fallback). Round 1 ~late Jun 2027, unverified. The call
    invites negative results. CVF counts a ≥20% overlap with an accepted paper as a duplicate.
  - **ICLR-27 workshops** (~1 Feb 2027): optional, non-archival, ≤4 pp, in person.
- **Skip:** AAAI-27 SAFE (2 Nov 2026, in person in Montréal, no virtual option); CVPR 2027 main
  (16 Nov 2026; its dual-submission window 16 Nov to 22 Feb blocks every workshop); AAAI-27
  workshops in general (in person, unfunded).
- **APSCON rule:** cite it in the third person, re-run rather than reuse its Florence FGSM/PGD
  results, and make new figures and text.

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

## Next Steps (2026-10-26)
Done:
- the N=5000 survey runs (all 3 tracks);
- the ECCV-W submission and its rejection;
- the PDF audit;
- the Phase-4 attack flags and fixes;
- the CPU rescore;
- `docs/paper_revision.md`.

**No GPU, no download:**
1. Apply `docs/paper_revision.md` to the LaTeX source: errata, ref [18] and [22], method
   rewrites, the RQ outline.
2. Import the N=5000 YOLO (Digvijay) and OCR (Ankush) `summary_{attack}.json` files and commit
   them. Check the 0.1311 Random Resize value.
3. Score-rule audit for Florence (`--score-rule {geo,random,inverse,constant}` in
   `p4_rescore.py`). Not written yet.
4. D0 decomposition and a worst-case aggregator over seeds and restarts, plus an L_inf assertion
   on saved adversarials. Not written yet.
5. Diff the draft against the APSCON PDF. Check whether EmergingAD's OpenReview made the rejected
   PDF public.

**Needs the user's go-ahead:**
6. Download val2017 (~780 MB). Every GPU run below needs it.

**GPU (an A6000 when available), in order:**
7. Timing, an eps-sanity check, and a 50-iteration convergence check at n=300.
8. clean_control: YOLO at n=5000; Florence det + OCR at n=1000.
9. adaptive_bpda on YOLO, n=1000 × 3 seeds. Run union-max vs mean-EOT on EnsJMG, then Median,
   then Ens4W, then an eps sweep.
10. Adaptive Florence-2 (new code) if time allows; otherwise scope the title to YOLO-World.

**Later:** BFG history cleanup before any public GitHub release.
