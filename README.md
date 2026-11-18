# Robustifying Zero-Shot Detection and OCR Against Adversarial Attacks

This repository studies whether input-transform defenses protect open-vocabulary models from
adversarial attacks. It covers four models, all evaluated on **COCO val2017**:

| Model | Type | Task scored |
|---|---|---|
| YOLOv8x-WorldV2 | one-stage open-vocabulary detector | detection, COCO mAP@[.5:.95] |
| Florence-2-Base | sequence-to-sequence VLM | detection (`<OD>`, mAP) and OCR (self-consistency) |
| SmolVLM-256M-Instruct | chat-style VLM (added in Phase 4) | yes/no object-presence questions, accuracy |
| OWL-ViT-B/32 | CLIP-based open-vocabulary detector (added 18 Nov) | detection, COCO mAP@[.5:.95] |

**Attacks:**
- Phase 3: FGSM, PGD and Patch.
- Phase 4: oblivious PGD, and an **adaptive BPDA+EOT PGD** that attacks through the defense.

**Defenses:**
- 13 single transforms: JPEG, Median, Gaussian, TVM, Blur+TVM, Bilateral, SVD, bit depth, dithering, NLM, anisotropic diffusion, BM3D and random resize.
- 7 detection-level ensembles that combine members by class-aware NMS: EnsJMG, EnsJMT, Ens4W, EnsBTC, EnsBMT, EnsJBT and EnsJTS.

**Where to find what:**
- Paper draft: `docs/paper_draft/main.tex`. Its `README.md` traces every number to a source file.
- Short write-up of the Phase-4 results: `docs/PHASE4_RESULTS_SUMMARY.md`.
- Every Phase-4 GPU job with its start, end, exit code and command: `docs/PHASE4_JOB_LEDGER.md`.
- Project state and citations: `docs/memory.md`.

## Key findings (19 Nov 2026)

1. **The Phase-3 numbers hold up.** The clean scores reproduce, and the oblivious defended scores land close to Phase 3's under a stronger PGD:
   - YOLO-World clean: 0.4519 on 5,000 images and 0.4764 on 1,000 (Phase 3: 0.4765).
   - Florence-2 clean: 0.3304 on 5,000 (Phase 3: 0.3300) and 0.3604 on 1,000 (Phase 3: 0.3605).
   - YOLO-World oblivious PGD + defense: Median 0.366 (Phase 3: 0.3639), EnsJMG 0.407 (Phase 3: 0.4045).
2. **Those recoveries only hold against an attacker who ignores the defense.** An adaptive attacker with the same 7/255 budget brings YOLO-World with a defense down to 0.016–0.069 mAP (worst case over objectives, seed 1), against 0.476 clean and 0.018 with no defense (seed 1; 0.015 over 3 seeds). On Florence-2 a defense keeps only +0.01 to +0.03 mAP over no defense (0.148 and 0.166 against 0.138 and 0.134).
3. **Ensembles are the least bad option, but not a fix.** Under the adaptive attack they keep 3–4 times more mAP than Median (0.048–0.069 vs 0.016). He et al. (WOOT 2017) found the same weakness for ensembles of weak defenses on image classifiers.
4. **Ensembles are cheap on clean detection** (at most 0.010 mAP lost on 5,000 images) **but expensive on OCR** (self-consistency 0.644–0.843).
5. **SmolVLM, oblivious attack only:** strong smoothing (Blur+TVM, random resize, BM3D) recovers 69–75% of the damage. The ensembles recover 8–24%.
6. **OWL-ViT (4th model) is where ensembles do help, against an attacker who ignores the defense.** Every ensemble beats its best member (by +0.003 to +0.009 mAP; 6 of 7 with 95% bootstrap intervals above zero), and no ensemble has a clean cost distinguishable from zero. Against an attacker who knows the defense, EnsJMG still falls from 0.201 to 0.061 mAP with the max objective (no defense: 0.011); on the first 300 images the mean objective goes lower still (0.073 vs 0.083).
7. **A smaller budget (4/255) leaves the defenses more, but the ensemble's lead stays small.** On YOLO-World, EnsJMG keeps 0.137 against 0.062 with no defense, only 0.011 more than JPEG alone (0.126). On Florence-2 it keeps 0.256 against 0.194–0.195.
8. **Correction of 19 Nov:** the adaptive Median number for YOLO-World (seed 1) is 0.0165, which is **0.016** at three decimals. Earlier drafts said 0.017 (rounded twice).

## Status

**Phase 3: complete.**
- v2 at N=1,000 covers all 9 tracks: 3 attacks × {YOLO detection, Florence detection, Florence OCR}.
- The survey at N=5,000 covers all three tracks. Only the Florence-2 detection survey files are in this repo.
- The N=5,000 YOLO and OCR *attacked* numbers come from the earlier paper's tables. Their summary files no longer exist.
- The N=5,000 YOLO *clean* baseline (0.4519) is now backed by a committed file: `results/phase4/clean_control_yolo/`.

**Phase 4: complete (18 Nov 2026).** 33 GPU jobs on one RTX 5000 Ada (32 GB), 0 failures:
- **Round 1, 17 Nov 03:34 to 18 Nov 03:10, 20 jobs:** adaptive attacks on YOLO-World and Florence-2 (1,000 images), the clean cost of every transform (5,000 images) and SmolVLM (1,000 images).
- **Round 2, 18 Nov 10:20 to 16:48, 13 jobs:** the 4/255 budget on YOLO-World and Florence-2 (300 images) and the fourth model, OWL-ViT (`phase4/p4_owl.py`, 1,000 images). The 2/255 budget was started and then dropped for time; nothing from it is cited.
- Every job ran with `--deterministic`, seed 1 unless it was a seed repeat, and the first N sorted val2017 images.
- **What is on GitHub:** every summary (`.json` + `.md`) and every run log, under `results/phase4/`; the per-job ledger `docs/PHASE4_JOB_LEDGER.md`.
- **What stays local:** the raw queue logs (`logs/phase4_queue/`), the resume checkpoints (`*.pkl`) and the per-image files. Every cited number is in a committed summary, including the per-image check (`frac_images_adaptive_below_oblivious`). Re-computing the first-300 re-scores or the OWL-ViT bootstrap from scratch needs the local checkpoints.

**Paper:** `docs/paper_draft/main.tex` is filled from `docs/paper_draft/make_phase4_tables.py`, which writes every Phase-4 table from the summaries below. `grep -n 'pending{' main.tex` lists anything still red.

**Warnings:**
- `results/results_survey_florence_detection/` (no attack suffix) is a **3-image smoke run**. Its `num_images` is 3 and its clean mAP 0.4998. It is not a result.
- Phase-3 outputs without `_v2` are historical and may include contamination from checkpoint reuse. Do not cite them.

## Results

**Rules for every table below:**
- **Every table between `<!-- table:... -->` markers is generated** from the committed result files by `scripts/make_readme_tables.py`. `python3 scripts/make_readme_tables.py --check` confirms that the README still matches the files. Do not edit those tables by hand.
- Detection is COCO mAP@[.5:.95].
- OCR is **self-consistency**: how closely the OCR text matches the model's *own* clean output. It is not accuracy.
- **N=1,000, N=300 and N=5,000 are different image sets** (the first N sorted val2017 images). Never compare numbers across them.
- Every table gives the actual scores. Changes and recoveries are extra columns next to them.
- Order: the Phase-4 tables first, then the earlier paper's Phase-3 tables in full.

### Ensembles

Every ensemble runs its members on the same image and merges their detections. For OCR the members vote; for SmolVLM their yes-probabilities are averaged.

<!-- table:ensembles -->
| Ensemble | Code name | Members (merged by class-aware NMS, IoU 0.5) | In the Phase-3 v2 runs |
|---|---|---|---|
| EnsJMG | `ens_jpeg_median_gaussian` | JPEG, Median, Gaussian | yes |
| EnsJMT | `ens_jpeg_median_tvm` | JPEG, Median, TVM | yes |
| Ens4W | `ens_4way` | JPEG, Median, TVM, Gaussian | no (added in the survey) |
| EnsBTC | `ens_blur_tvm_combo` | JPEG, Blur+TVM, Median | yes |
| EnsBMT | `ens_bilateral_median` | Bilateral, Median, TVM | no (added in the survey) |
| EnsJBT | `ens_jpeg_bilateral_tvm` | JPEG, Bilateral, TVM | no (added in the survey) |
| EnsJTS | `ens_jpeg_tvm_svd` | JPEG, TVM, SVD | no (added in the survey) |
<!-- /table:ensembles -->

### Phase 4 — adaptive attack on YOLO-World (N=1,000)

**Common settings:**
- PGD with eps 0.03, which is at most 7/255 per pixel at native resolution.
- 50 steps, alpha 0.002, `--attack-space native`, `--pgd-select best`.

**How the oblivious and adaptive attacks differ:**
- The oblivious attack ignores the defense; the defense is then applied to its image.
- The adaptive attack differentiates through the defense: BPDA (identity gradient) for JPEG, the true gradient for Median and Gaussian, and EOT over the ensemble branches.
- Both attacks start from the same seeded point.

**Columns:**
- *Max objective* raises the loss of the least-damaged branch; *mean objective* averages the branches. A single transform has one branch, so the two are the same.
- *Worst case* is the lower of the two objectives (seed 1).
- *Adaptive image (max objective), defense removed* scores the max-objective adaptive image on the bare model. It stays high for Median: the perturbation is shaped for the defense.
- Cells over several seeds give the mean, with the standard deviation in parentheses. Every single run is listed in the raw-values table further down.

<!-- table:p4_yolo_adaptive -->
| Defense | Seeds | Clean, no defense | Clean + defense | Oblivious PGD, no defense | Oblivious PGD + defense | Adaptive + defense, max objective | Adaptive + defense, mean objective (seed 1) | Worst case (seed 1) | Adaptive image (max objective), defense removed |
|---|---|---|---|---|---|---|---|---|---|
| No defense | 1, 2, 3 | 0.476 |  | 0.015 (sd 0.003) |  |  |  |  |  |
| Median | 1, 2, 3 | 0.476 | 0.458 | 0.015 (sd 0.003) | 0.366 (sd 0.003) | 0.018 (sd 0.001) | same (one branch) | **0.016** | 0.452 (sd 0.002) |
| EnsJMG | 1, 2, 3 | 0.476 | 0.466 | 0.015 (sd 0.003) | 0.407 (sd 0.004) | 0.073 (sd 0.002) | 0.048 | **0.048** | 0.107 (sd 0.003) |
| Ens4W | 1 | 0.476 | 0.468 | 0.018 | 0.431 | 0.108 | 0.069 | **0.069** | 0.083 |
<!-- /table:p4_yolo_adaptive -->

Source: `results/phase4/adaptive_bpda_yolo/summary_adaptive_bpda_yolo_<target>_seed<S>[_eot-loss-mean]_attack-space-native_pgd-select-best_latest.json`.

### Phase 4 — adaptive attack on Florence-2 detection (N=1,000, seed 1)

The attack works in pixel space at 7/255. It uses token cross-entropy against the model's own clean decode, with the same PGD loop and surrogates as YOLO-World. The mean objective was run on the first 300 images only, where it is within 0.002 of the max objective (0.201 vs 0.199, see the checks), so the EnsJMG worst case here is the max objective only.

<!-- table:p4_florence_adaptive -->
| Defense | Seeds | Clean, no defense | Clean + defense | Oblivious PGD, no defense | Oblivious PGD + defense | Adaptive + defense, max objective | Adaptive + defense, mean objective (seed 1) | Worst case (seed 1) | Adaptive image (max objective), defense removed |
|---|---|---|---|---|---|---|---|---|---|
| No defense | 1 | 0.360 |  | 0.138 |  |  |  |  |  |
| Median | 1 | 0.360 | 0.350 | 0.138 | 0.256 | 0.148 | same (one branch) | **0.148** | 0.335 |
| EnsJMG | 1 | 0.360 | 0.353 | 0.134 | 0.264 | 0.166 | not run | **0.166** (max only) | 0.159 |
<!-- /table:p4_florence_adaptive -->

Florence-2 is not bit-repeatable, so the no-defense columns differ slightly between the two runs. Source: `results/phase4/adaptive_bpda_florence/summary_adaptive_bpda_florence_<target>_seed1_pgd-select-best_latest.json`.

### Phase 4 — checks that the attack is sound

<!-- table:p4_checks -->
| Check | Setting | Result | Meaning |
|---|---|---|---|
| Adaptive vs oblivious objective | all 14 runs on 1,000 images (3 models) | the adaptive objective is below the oblivious one on 999–1,000 of 1,000 images, and on all 300 images of every 300-image run | the optimizer does what it should |
| More steps | YOLO-World EnsJMG, 300 images | 50 steps 0.086 → 100 steps 0.054 | not converged at 50 steps: the adaptive numbers are **upper bounds** |
| Larger budget | YOLO-World EnsJMG, eps 0.1 (alpha 0.0067), 300 images | adaptive + defense 0.023 (oblivious + defense 0.340) | mAP keeps falling as eps grows: no sign of gradient masking |
| Max vs mean objective | YOLO-World EnsJMG and Ens4W (1,000 images); Florence-2 and OWL-ViT EnsJMG (first 300) | YOLO-World: mean is stronger (0.071 → 0.048, 0.108 → 0.069). OWL-ViT: mean is stronger (0.083 → 0.073). Florence-2: within 0.002 (0.199 max vs 0.201 mean) | report the **worst case over objectives** |
| Median gradient | YOLO-World Median, true vs identity (BPDA), 1,000 images | 0.016 vs 0.042 | the true median gradient is the stronger attack |
| Repeatability | same seed and images, `--deterministic`: oblivious PGD with no defense, across the runs that compute it | YOLO-World 0.0183 in all 6 runs (identical); Florence-2 0.1336–0.1381 over 2 runs; OWL-ViT 0.0108–0.0117 over 3 runs | YOLO-World is bit-repeatable; Florence-2 and OWL-ViT are not |
<!-- /table:p4_checks -->

Sources: every run in the raw-values table below, in particular `results/phase4/adaptive_bpda_yolo_iters50_n300/`, `adaptive_bpda_yolo_iters100_n300/`, `adaptive_bpda_yolo_eps0.1/`, `adaptive_bpda_florence_n300/`, `adaptive_bpda_owl_n300/`, and the `_median-surrogate-identity` and `_eot-loss-mean` summaries in `adaptive_bpda_yolo/`.

**Note on the 300-image numbers:** the first 300 images of a 1,000-image run are the same images. `scripts/p4_subset_score.py` scores them from the 1,000-image checkpoint (`summary_subset_*_first300.json`). On YOLO-World it reproduces a separate 300-image run exactly.

### Phase 4 — clean cost of every transform (N=5,000, no attack)

Each transform applied to clean images, against the same model with no transform. *change* = with the transform minus without.

<!-- table:p4_clean_cost -->
| Defense | YOLO-World mAP | change | Florence-2 det. mAP | change | OCR self-consistency | change |
|---|---|---|---|---|---|---|
| **No defense** | 0.4519 |  | 0.3304 |  | 1.000 |  |
| EnsJTS | 0.4522 | +0.000 | 0.3238 | −0.007 | 0.843 | −0.157 |
| Ens4W | 0.4480 | −0.004 | 0.3212 | −0.009 | 0.746 | −0.254 |
| EnsJBT | 0.4480 | −0.004 | 0.3253 | −0.005 | 0.789 | −0.211 |
| EnsBMT | 0.4477 | −0.004 | 0.3258 | −0.005 | 0.788 | −0.212 |
| EnsJMG | 0.4468 | −0.005 | 0.3242 | −0.006 | 0.713 | −0.287 |
| EnsJMT | 0.4466 | −0.005 | 0.3241 | −0.006 | 0.746 | −0.254 |
| EnsBTC | 0.4424 | −0.010 | 0.3202 | −0.010 | 0.644 | −0.356 |
| SVD | 0.4519 | +0.000 | 0.3284 | −0.002 | 0.953 | −0.047 |
| NLM | 0.4503 | −0.002 | 0.3107 | −0.020 | 0.886 | −0.114 |
| Bilateral | 0.4463 | −0.006 | 0.3314 | +0.001 | 0.821 | −0.179 |
| Bit depth | 0.4448 | −0.007 | 0.2802 | −0.050 | 0.797 | −0.203 |
| Dithering | 0.4431 | −0.009 | 0.3036 | −0.027 | 0.776 | −0.224 |
| Anisotropic diffusion | 0.4399 | −0.012 | 0.3163 | −0.014 | 0.798 | −0.202 |
| Median | 0.4387 | −0.013 | 0.3202 | −0.010 | 0.620 | −0.380 |
| Gaussian | 0.4369 | −0.015 | 0.2744 | −0.056 | 0.672 | −0.328 |
| TVM | 0.4280 | −0.024 | 0.2643 | −0.066 | 0.725 | −0.275 |
| JPEG | 0.4262 | −0.026 | 0.3198 | −0.011 | 0.673 | −0.327 |
| Blur+TVM | 0.3953 | −0.057 | 0.2492 | −0.081 | 0.507 | −0.493 |
| BM3D |  |  |  |  | 0.648 | −0.352 |
| Random resize |  |  |  |  | 0.707 | −0.293 |
<!-- /table:p4_clean_cost -->

Sources: `results/phase4/clean_control_{yolo,florence_det,florence_ocr}/summary_clean_control_*_alltransforms_latest.json`. BM3D and random resize were run for OCR only.

The OCR column bounds OCR recovery: a transform cannot restore more self-consistency under attack than it keeps on clean images.

### Phase 4 — SmolVLM-256M yes/no presence questions (N=1,000, 2,000 questions)

**Setup:**
- Each image is resized to a 512×512 square in every condition.
- Questions are "Is there a {class} in this image?", two per image; 992 of the 2,000 have the answer yes.
- Attack: oblivious PGD, eps 0.03, 10 steps, alpha 0.0075.
- An ensemble answers with the mean of its members' yes-probabilities.
- Clean cost = accuracy with the transform minus without, on clean images. Recovery = attacked accuracy with the transform minus without. Share of damage recovered = recovery / (clean − attacked accuracy). Brackets are 95% paired bootstrap intervals; **bold** marks an interval that excludes zero.

<!-- table:p4_smolvlm -->
| Defense | Clean accuracy | Clean cost [95% CI] | Attacked accuracy | Recovery [95% CI] | Share of damage recovered |
|---|---|---|---|---|---|
| **No defense** | 0.856 |  | 0.439 |  |  |
| Blur+TVM | 0.786 | **−0.070 [−0.084, −0.057]** | 0.754 | **+0.315 [+0.297, +0.335]** | 75% |
| Random resize | 0.825 | **−0.031 [−0.043, −0.019]** | 0.748 | **+0.309 [+0.292, +0.328]** | 74% |
| BM3D | 0.810 | **−0.046 [−0.059, −0.033]** | 0.728 | **+0.290 [+0.272, +0.309]** | 69% |
| TVM | 0.832 | **−0.025 [−0.037, −0.013]** | 0.638 | **+0.200 [+0.182, +0.218]** | 48% |
| Gaussian | 0.820 | **−0.037 [−0.049, −0.024]** | 0.580 | **+0.142 [+0.128, +0.159]** | 34% |
| EnsBTC | 0.831 | **−0.026 [−0.036, −0.015]** | 0.538 | **+0.100 [+0.087, +0.113]** | 24% |
| Ens4W | 0.835 | **−0.021 [−0.031, −0.011]** | 0.532 | **+0.093 [+0.081, +0.105]** | 22% |
| Median | 0.845 | **−0.012 [−0.022, −0.001]** | 0.531 | **+0.093 [+0.081, +0.107]** | 22% |
| EnsJMG | 0.834 | **−0.022 [−0.032, −0.012]** | 0.523 | **+0.084 [+0.072, +0.097]** | 20% |
| EnsJMT | 0.839 | **−0.018 [−0.028, −0.008]** | 0.522 | **+0.083 [+0.071, +0.096]** | 20% |
| EnsBMT | 0.843 | **−0.014 [−0.023, −0.004]** | 0.507 | **+0.069 [+0.057, +0.081]** | 16% |
| JPEG | 0.827 | **−0.030 [−0.041, −0.018]** | 0.483 | **+0.044 [+0.034, +0.054]** | 11% |
| EnsJBT | 0.837 | **−0.019 [−0.030, −0.009]** | 0.479 | **+0.040 [+0.031, +0.050]** | 10% |
| EnsJTS | 0.846 | **−0.010 [−0.018, −0.002]** | 0.470 | **+0.032 [+0.024, +0.041]** | 8% |
| Anisotropic diffusion | 0.846 | **−0.011 [−0.021, −0.001]** | 0.465 | **+0.026 [+0.017, +0.034]** | 6% |
| Bilateral | 0.842 | **−0.014 [−0.024, −0.004]** | 0.453 | **+0.014 [+0.007, +0.021]** | 3% |
| NLM | 0.853 | −0.004 [−0.009, +0.002] | 0.441 | +0.003 [−0.003, +0.009] | 1% |
| Dithering | 0.853 | −0.004 [−0.012, +0.005] | 0.439 | +0.001 [−0.004, +0.004] | 0% |
| SVD | 0.855 | −0.001 [−0.004, +0.001] | 0.439 | +0.000 [+0.000, +0.000] | 0% |
| Bit depth | 0.850 | −0.006 [−0.016, +0.004] | 0.438 | −0.001 [−0.005, +0.003] | 0% |
<!-- /table:p4_smolvlm -->

Source: `results/phase4/presence_qa_smolvlm/summary_presence_qa_smolvlm_latest.json`.

We did not run an adaptive attack on SmolVLM. Given the YOLO-World and Florence-2 results, read these recoveries as upper bounds.

### Phase 4 — fourth model: OWL-ViT-B/32 (N=1,000)

**Setup:**
- Model: `google/owlvit-base-patch32`, about 150M parameters, prompted with "a photo of a {class}" for the 80 COCO classes.
- Detections: score ≥ 0.001, class-aware NMS at 0.5, at most 300 per image (YOLO-World's settings).
- Attack: the same objective as YOLO-World's (sum over patches of the max-class score), pixel space, 7/255, 50 steps, alpha 0.002.

**Under the attack that ignores the defense** (`oblivious_survey`, every transform in one run):
- Clean cost = clean mAP with the transform minus without. Recovery = attacked mAP with the transform minus without. Share of damage recovered = recovery / (clean − attacked).
- The last column is the ensemble's attacked mAP minus that of its best member under the same attack, with a 95% paired bootstrap interval (1,000 resamples). **Bold** marks an interval that excludes zero.

<!-- table:p4_owl_survey -->
| Defense | Clean + defense | Clean cost (ensembles: [95% CI]) | Attacked + defense | Recovery | Share of damage recovered | Ensemble − best member, attacked [95% CI] (member) |
|---|---|---|---|---|---|---|
| **No defense** | 0.310 |  | 0.012 |  |  |  |
| EnsBTC | 0.311 | +0.002 [−0.006, +0.010] | 0.243 | +0.232 | 78% | **+0.007 [+0.004, +0.013]** (Blur+TVM) |
| Ens4W | 0.314 | +0.005 [−0.002, +0.012] | 0.237 | +0.225 | 75% | **+0.008 [+0.004, +0.013]** (TVM) |
| Blur+TVM | 0.268 | −0.042 | 0.236 | +0.224 | 75% |  |
| EnsJMT | 0.313 | +0.003 [−0.004, +0.010] | 0.233 | +0.222 | 74% | **+0.004 [+0.001, +0.009]** (TVM) |
| EnsJBT | 0.314 | +0.004 [−0.003, +0.010] | 0.233 | +0.221 | 74% | **+0.004 [+0.001, +0.007]** (TVM) |
| EnsJTS | 0.316 | +0.006 [−0.000, +0.011] | 0.232 | +0.221 | 74% | **+0.003 [+0.001, +0.007]** (TVM) |
| EnsBMT | 0.313 | +0.003 [−0.005, +0.009] | 0.231 | +0.220 | 74% | +0.003 [−0.000, +0.006] (TVM) |
| TVM | 0.295 | −0.015 | 0.229 | +0.217 | 73% |  |
| EnsJMG | 0.315 | +0.005 [−0.001, +0.012] | 0.201 | +0.189 | 63% | **+0.009 [+0.005, +0.015]** (Gaussian) |
| Gaussian | 0.297 | −0.013 | 0.192 | +0.180 | 60% |  |
| Median | 0.304 | −0.006 | 0.168 | +0.157 | 53% |  |
| JPEG | 0.304 | −0.006 | 0.167 | +0.156 | 52% |  |
| Anisotropic diffusion | 0.305 | −0.005 | 0.146 | +0.134 | 45% |  |
| Bilateral | 0.306 | −0.004 | 0.139 | +0.127 | 43% |  |
| NLM | 0.310 | −0.000 | 0.114 | +0.103 | 34% |  |
| Bit depth | 0.301 | −0.009 | 0.050 | +0.038 | 13% |  |
| Dithering | 0.299 | −0.011 | 0.047 | +0.035 | 12% |  |
| SVD | 0.311 | +0.001 | 0.012 | +0.001 | 0% |  |
<!-- /table:p4_owl_survey -->

Every ensemble beats its best member (6 of 7 intervals exclude zero), and no ensemble's clean-cost interval excludes zero.

**Under the attack that knows the defense** (`adaptive_bpda`):

<!-- table:p4_owl_adaptive -->
| Target | Objective | Images | Clean, no defense | Clean + defense | Oblivious PGD, no defense | Oblivious PGD + defense | **Adaptive + defense** | Adaptive image, defense removed |
|---|---|---|---|---|---|---|---|---|
| Median | (one branch) | 1,000 | 0.310 | 0.304 | 0.011 | 0.169 | **0.021** | 0.244 |
| EnsJMG | max | 1,000 | 0.310 | 0.315 | 0.011 | 0.201 | **0.061** | 0.041 |
| EnsJMG | mean | 300 | 0.323 | 0.337 | 0.012 | 0.223 | **0.073** | 0.046 |
| EnsJMG | max | 300 (first 300 of the 1,000-image run) | 0.323 | 0.337 | 0.012 | 0.223 | **0.083** | 0.063 |
<!-- /table:p4_owl_adaptive -->

The adaptive objective was below the oblivious one on every image of every run. OWL-ViT is not bit-repeatable: the oblivious no-defense number is 0.0108–0.0117 across the three 1,000-image runs that compute it (the two adaptive runs and the survey). EnsJMG at 1,000 images was run with the max objective only; on the first 300 images the mean objective went lower (0.073 vs 0.083), so 0.061 is not the worst case.

Sources:
- `results/phase4/oblivious_survey_owl/`: the survey summary, plus `summary_bootstrap_oblivious_survey_owl_seed1.json` from `scripts/p4_boot_survey.py`.
- `results/phase4/adaptive_bpda_owl/` and `results/phase4/adaptive_bpda_owl_n300/`.

### Phase 4 — attack budget, 4/255 vs 7/255 (first 300 images, seed 1)

Defended mAP under the oblivious and the adaptive attack, at two budgets, on the same 300 images.

<!-- table:p4_budget -->
| Model | Defense | Objective | Clean + defense | 4/255 oblivious + defense | **4/255 adaptive + defense** | 7/255 oblivious + defense | **7/255 adaptive + defense** |
|---|---|---|---|---|---|---|---|
| YOLO-World | no defense |  | 0.503 | 0.062 | (same) | 0.017 | (same) |
|  | Median | (one branch) | 0.489 | 0.430 | **0.054** | 0.402 | **0.022** |
|  | JPEG | (one branch) | 0.487 | 0.425 | **0.126** | 0.380 | **0.062** |
|  | Gaussian | (one branch) | 0.494 | 0.461 | **0.082** | 0.443 | **0.035** |
|  | EnsJMG | max | 0.500 | 0.469 | **0.174** | 0.441 | **0.086** |
|  | EnsJMG | mean | 0.500 | 0.469 | **0.137** | 0.441 | **0.066** |
| Florence-2 | no defense |  | 0.391 | 0.194–0.195 (2 runs) | (same) | 0.167–0.172 (3 runs) | (same) |
|  | Median | (one branch) | 0.390 | 0.327 | **0.214** | 0.293 | **0.165** |
|  | EnsJMG | max | 0.386 | 0.332 | **0.256** | 0.300 | **0.199** |
|  | EnsJMG | mean | 0.386 | not run | not run | 0.302 | **0.201** |
<!-- /table:p4_budget -->

Notes:
- **JPEG is the member whose attack gradient is approximated (BPDA identity),** so its adaptive numbers are the most generous here.
- **The worst case is the lower of the two EnsJMG rows.** On YOLO-World that is the mean objective at both budgets (0.137 and 0.066). On Florence-2 only the max objective was run at 4/255; at 7/255 the two are within 0.002.
- **Where two values are given**, the runs that compute them differ because Florence-2 is not bit-repeatable. YOLO-World runs agree bit for bit.
- Sources: `results/phase4/adaptive_bpda_{yolo,florence}_eps4of255_n300/`, `adaptive_bpda_yolo_eps7of255_n300/` (JPEG and Gaussian), `adaptive_bpda_florence_n300/` (Florence-2 EnsJMG mean), and `budget_curve_eps7of255_first300/` (the other 7/255 rows, scored from the 1,000-image runs).

### Phase 4 — every adaptive-attack run (raw values)

Every adaptive summary in `results/phase4/`, at 4 decimals. Columns as above; *Adaptive objective below oblivious* counts the images on which the adaptive attack reached a lower objective than the oblivious one (not stored for the first-300 re-scores).

<!-- table:p4_all_adaptive -->
| Model | Target | Seed | Objective | Budget | Steps | Images | Note | Clean | Clean + def. | Obl., no def. | Obl. + def. | Adapt. image, def. removed | Adapt. + def. | Adaptive objective below oblivious | Directory (`results/phase4/`) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO-World | Median | 1 | (one branch) | 7/255 | 50 | 1,000 | main setting | 0.4764 | 0.4583 | 0.0183 | 0.3691 | 0.4520 | **0.0165** | 1000/1000 | `adaptive_bpda_yolo` |
| YOLO-World | Median | 2 | (one branch) | 7/255 | 50 | 1,000 | main setting | 0.4764 | 0.4583 | 0.0149 | 0.3669 | 0.4511 | **0.0181** | 1000/1000 | `adaptive_bpda_yolo` |
| YOLO-World | Median | 3 | (one branch) | 7/255 | 50 | 1,000 | main setting | 0.4764 | 0.4583 | 0.0131 | 0.3623 | 0.4542 | **0.0191** | 999/1000 | `adaptive_bpda_yolo` |
| YOLO-World | Median | 1 | (one branch) | 7/255 | 50 | 1,000 | median gradient: identity | 0.4764 | 0.4583 | 0.0183 | 0.3691 | 0.2721 | **0.0420** | 1000/1000 | `adaptive_bpda_yolo` |
| YOLO-World | EnsJMG | 1 | max | 7/255 | 50 | 1,000 | main setting | 0.4764 | 0.4656 | 0.0183 | 0.4106 | 0.1033 | **0.0713** | 1000/1000 | `adaptive_bpda_yolo` |
| YOLO-World | EnsJMG | 2 | max | 7/255 | 50 | 1,000 | main setting | 0.4764 | 0.4656 | 0.0149 | 0.4068 | 0.1082 | **0.0746** | 1000/1000 | `adaptive_bpda_yolo` |
| YOLO-World | EnsJMG | 3 | max | 7/255 | 50 | 1,000 | main setting | 0.4764 | 0.4656 | 0.0131 | 0.4029 | 0.1082 | **0.0741** | 1000/1000 | `adaptive_bpda_yolo` |
| YOLO-World | EnsJMG | 1 | mean | 7/255 | 50 | 1,000 | main setting | 0.4764 | 0.4656 | 0.0183 | 0.4106 | 0.0817 | **0.0484** | 1000/1000 | `adaptive_bpda_yolo` |
| YOLO-World | Ens4W | 1 | max | 7/255 | 50 | 1,000 | main setting | 0.4764 | 0.4684 | 0.0183 | 0.4312 | 0.0828 | **0.1075** | 1000/1000 | `adaptive_bpda_yolo` |
| YOLO-World | Ens4W | 1 | mean | 7/255 | 50 | 1,000 | main setting | 0.4764 | 0.4684 | 0.0183 | 0.4312 | 0.0518 | **0.0687** | 1000/1000 | `adaptive_bpda_yolo` |
| YOLO-World | JPEG | 1 | (one branch) | 7/255 | 50 | 300 | main setting | 0.5032 | 0.4867 | 0.0175 | 0.3804 | 0.1725 | **0.0621** | 300/300 | `adaptive_bpda_yolo_eps7of255_n300` |
| YOLO-World | Gaussian | 1 | (one branch) | 7/255 | 50 | 300 | main setting | 0.5032 | 0.4935 | 0.0175 | 0.4430 | 0.2860 | **0.0351** | 300/300 | `adaptive_bpda_yolo_eps7of255_n300` |
| YOLO-World | EnsJMG | 1 | max | 7/255 | 50 | 300 | main setting | 0.5032 | 0.4995 | 0.0175 | 0.4407 | 0.1160 | **0.0859** | 300/300 | `adaptive_bpda_yolo_iters50_n300` |
| YOLO-World | EnsJMG | 1 | max | 7/255 | 100 | 300 | 100 steps | 0.5032 | 0.4995 | 0.0033 | 0.4428 | 0.0604 | **0.0536** | 300/300 | `adaptive_bpda_yolo_iters100_n300` |
| YOLO-World | Median | 1 | (one branch) | 7/255 | 50 | 300 | first 300 of the 1,000-image run | 0.5032 | 0.4891 | 0.0175 | 0.4022 | 0.4841 | **0.0222** |  | `budget_curve_eps7of255_first300` |
| YOLO-World | EnsJMG | 1 | max | 7/255 | 50 | 300 | first 300 of the 1,000-image run | 0.5032 | 0.4995 | 0.0175 | 0.4407 | 0.1160 | **0.0859** |  | `budget_curve_eps7of255_first300` |
| YOLO-World | EnsJMG | 1 | mean | 7/255 | 50 | 300 | first 300 of the 1,000-image run | 0.5032 | 0.4995 | 0.0175 | 0.4407 | 0.0961 | **0.0661** |  | `budget_curve_eps7of255_first300` |
| YOLO-World | Median | 1 | (one branch) | 4/255 | 50 | 300 | 4/255 | 0.5032 | 0.4891 | 0.0619 | 0.4301 | 0.4964 | **0.0536** | 300/300 | `adaptive_bpda_yolo_eps4of255_n300` |
| YOLO-World | JPEG | 1 | (one branch) | 4/255 | 50 | 300 | 4/255 | 0.5032 | 0.4867 | 0.0619 | 0.4252 | 0.2786 | **0.1260** | 300/300 | `adaptive_bpda_yolo_eps4of255_n300` |
| YOLO-World | Gaussian | 1 | (one branch) | 4/255 | 50 | 300 | 4/255 | 0.5032 | 0.4935 | 0.0619 | 0.4605 | 0.3420 | **0.0822** | 300/300 | `adaptive_bpda_yolo_eps4of255_n300` |
| YOLO-World | EnsJMG | 1 | max | 4/255 | 50 | 300 | 4/255 | 0.5032 | 0.4995 | 0.0619 | 0.4688 | 0.2243 | **0.1738** | 300/300 | `adaptive_bpda_yolo_eps4of255_n300` |
| YOLO-World | EnsJMG | 1 | mean | 4/255 | 50 | 300 | 4/255 | 0.5032 | 0.4995 | 0.0619 | 0.4688 | 0.1767 | **0.1368** | 300/300 | `adaptive_bpda_yolo_eps4of255_n300` |
| YOLO-World | EnsJMG | 1 | max | eps 0.1 | 50 | 300 | eps 0.1 | 0.5032 | 0.4995 | 0.0019 | 0.3399 | 0.0298 | **0.0229** | 300/300 | `adaptive_bpda_yolo_eps0.1` |
| Florence-2 | Median | 1 | (one branch) | 7/255 | 50 | 1,000 | main setting | 0.3604 | 0.3504 | 0.1381 | 0.2561 | 0.3346 | **0.1480** | 1000/1000 | `adaptive_bpda_florence` |
| Florence-2 | EnsJMG | 1 | max | 7/255 | 50 | 1,000 | main setting | 0.3604 | 0.3529 | 0.1336 | 0.2642 | 0.1588 | **0.1656** | 1000/1000 | `adaptive_bpda_florence` |
| Florence-2 | EnsJMG | 1 | mean | 7/255 | 50 | 300 | main setting | 0.3910 | 0.3860 | 0.1715 | 0.3017 | 0.2128 | **0.2014** | 300/300 | `adaptive_bpda_florence_n300` |
| Florence-2 | Median | 1 | (one branch) | 7/255 | 50 | 300 | first 300 of the 1,000-image run | 0.3910 | 0.3896 | 0.1709 | 0.2934 | 0.3546 | **0.1651** |  | `budget_curve_eps7of255_first300` |
| Florence-2 | EnsJMG | 1 | max | 7/255 | 50 | 300 | first 300 of the 1,000-image run | 0.3910 | 0.3860 | 0.1669 | 0.3001 | 0.1884 | **0.1993** |  | `adaptive_bpda_florence_n300` |
| Florence-2 | Median | 1 | (one branch) | 4/255 | 50 | 300 | 4/255 | 0.3910 | 0.3896 | 0.1954 | 0.3271 | 0.3809 | **0.2142** | 300/300 | `adaptive_bpda_florence_eps4of255_n300` |
| Florence-2 | EnsJMG | 1 | max | 4/255 | 50 | 300 | 4/255 | 0.3910 | 0.3860 | 0.1941 | 0.3323 | 0.2319 | **0.2555** | 300/300 | `adaptive_bpda_florence_eps4of255_n300` |
| OWL-ViT | Median | 1 | (one branch) | 7/255 | 50 | 1,000 | main setting | 0.3099 | 0.3043 | 0.0108 | 0.1687 | 0.2444 | **0.0213** | 1000/1000 | `adaptive_bpda_owl` |
| OWL-ViT | EnsJMG | 1 | max | 7/255 | 50 | 1,000 | main setting | 0.3099 | 0.3150 | 0.0114 | 0.2010 | 0.0408 | **0.0610** | 1000/1000 | `adaptive_bpda_owl` |
| OWL-ViT | EnsJMG | 1 | mean | 7/255 | 50 | 300 | main setting | 0.3231 | 0.3371 | 0.0123 | 0.2232 | 0.0459 | **0.0728** | 300/300 | `adaptive_bpda_owl_n300` |
| OWL-ViT | EnsJMG | 1 | max | 7/255 | 50 | 300 | first 300 of the 1,000-image run | 0.3231 | 0.3371 | 0.0124 | 0.2229 | 0.0628 | **0.0829** |  | `adaptive_bpda_owl_n300` |
<!-- /table:p4_all_adaptive -->

### How Phase 4 changes the Phase-3 conclusions

<!-- table:p4_vs_p3 -->
| YOLO-World, same 1,000 images | Phase 3 (v2) | Phase 4, oblivious | Phase 4, adaptive (worst case) |
|---|---|---|---|
| Attack | PGD, 10 steps, alpha 0.0075, letterbox space | PGD, 50 steps, alpha 0.002, native resolution | BPDA+EOT through the defense, 50 steps |
| Clean, no defense | 0.4765 | 0.476 | 0.476 |
| Attacked, no defense | 0.0737 | 0.015 (3 seeds) | (no defense to adapt to) |
| Median | 0.3639 | 0.366 (3 seeds) | 0.016 (seed 1) |
| EnsJMG | 0.4045 | 0.407 (3 seeds) | 0.048 (seed 1) |
| Ens4W | not in v2 | 0.431 (seed 1) | 0.069 (seed 1) |
| EnsBTC | 0.4128 | not run | not run |
<!-- /table:p4_vs_p3 -->

- **The clean scores reproduce Phase 3** (0.476 vs 0.4765), so the code and data agree with the paper. The oblivious defended scores land close to Phase 3's (Median 0.366 vs 0.3639, EnsJMG 0.407 vs 0.4045) although the Phase-4 PGD is a different, stronger attack: it hurts the undefended model more (0.015 vs 0.074). That makes the defenses look slightly *better* under the Phase-3 threat model.
- **The change is the threat model.** Once the attacker knows the defense, the Phase-3 recovery of Median and EnsJMG (+0.29 to +0.33 mAP) disappears. That includes the ensembles the Phase-3 paper recommended.
- **What the paper must change:**
  - Present the ensemble result two-sided: it helps against an unaware attacker, and little survives an aware one.
  - Present the adaptive numbers as upper bounds (the attack had not converged at 50 steps).
  - Report the worst case over the objectives.
  - Drop the claim that the "max" objective is the natural worst case, and the "union-aware" novelty claim. On YOLO-World and OWL-ViT the mean objective was stronger.
- **Florence-2 Phase-3 PGD numbers are not comparable to Phase 4.** Phase 3 attacked Florence-2 in normalized space (about 1.7/255 per pixel); Phase 4 attacks it at 7/255 in pixel space.

### Phase 3 — Locked v2, N=1,000 (the earlier paper's main table)

The scripts in the *Generated by* column now live in `archive/Phase3/`. Recovery = (defended − attacked) / (clean − attacked), as stored in each file.

<!-- table:p3_v2_best -->
| Track | Clean | Attacked | Best defense | Defended | Recovery | Generated by | Source file |
|---|---|---|---|---|---|---|---|
| YOLO-World · FGSM | 0.4765 | 0.2259 | Blur+TVM | **0.3448** | +47.4% | `FGSM_Phase3_YOLO_v2.ipynb` | `results/results_phase3_yolo_fgsm_v2/summary.json` |
| YOLO-World · PGD (10 steps) | 0.4765 | 0.0737 | EnsBTC | **0.4128** | +84.2% | `PGD_Phase3_YOLO_v2.ipynb` | `results/results_phase3_yolo_pgd_v2/summary.json` |
| YOLO-World · Patch | 0.4765 | 0.4537 | EnsJMT | **0.4599** | +27.5% | `Patch_Phase3_YOLO_v2.ipynb` | `results/results_phase3_yolo_patch_v2/summary.json` |
| Florence-2 det. · FGSM | 0.3605 | 0.2731 | EnsJMG | **0.3281** | +62.9% | `FGSM_Phase3_Florence_v2.py` | `results/results_phase3_florence_fgsm_v2/summary.json` |
| Florence-2 det. · PGD | 0.3605 | 0.2243 | EnsJMG | **0.3245** | +73.5% | `PGD_Phase3_Florence_v2.py` | `results/results_phase3_florence_pgd_v2/summary.json` |
| Florence-2 det. · Patch | 0.3605 | 0.3268 | EnsBTC | **0.3324** | +16.7% | `Patch_Phase3_Florence_v2.py` | `results/results_phase3_florence_patch_v2/summary.json` |
| Florence-2 OCR · FGSM | 1.0000 | 0.3287 | TVM | **0.5615** | +34.7% | `FGSM_Florence2_OCR_Robust.py` | `results/results_fgsm_florence2_ocr_robust/summary.json` |
| Florence-2 OCR · PGD (alpha 0.003) | 1.0000 | 0.3139 | TVM | **0.6384** | +47.3% | `PGD_Florence2_OCR_Robust.py` | `results/results_pgd_florence2_ocr_robust/summary.json` |
| Florence-2 OCR · Patch | 1.0000 | 0.3675 | Median | **0.5566** | +29.9% | `Patch_Florence2_OCR_Robust.py` | `results/results_patch_florence2_ocr_robust/summary.json` |
<!-- /table:p3_v2_best -->

Every defense on every v2 track. Each cell is the defended score, with the recovery in parentheses; **bold** is the best in its column. Detection is mAP, OCR is self-consistency.

<!-- table:p3_v2_full -->
|  | YOLO-World · FGSM | YOLO-World · PGD | YOLO-World · Patch | Florence-2 det. · FGSM | Florence-2 det. · PGD | Florence-2 det. · Patch | Florence-2 OCR · FGSM | Florence-2 OCR · PGD | Florence-2 OCR · Patch |
|---|---|---|---|---|---|---|---|---|---|
| Clean, no defense | 0.4765 | 0.4765 | 0.4765 | 0.3605 | 0.3605 | 0.3605 | 1.0000 | 1.0000 | 1.0000 |
| Attacked, no defense | 0.2259 | 0.0737 | 0.4537 | 0.2731 | 0.2243 | 0.3268 | 0.3287 | 0.3139 | 0.3675 |
| JPEG | 0.2697 (+17%) | 0.3643 (+72%) | 0.4391 (−64%) | 0.3199 (+54%) | 0.3162 (+67%) | 0.3270 (+1%) | 0.4963 (+25%) | 0.6083 (+43%) | 0.3765 (+1%) |
| Median | 0.2928 (+27%) | 0.3639 (+72%) | 0.4496 (−18%) | 0.3194 (+53%) | 0.3198 (+70%) | 0.3289 (+6%) | 0.4275 (+15%) | 0.5409 (+33%) | **0.5566 (+30%)** |
| TVM | 0.2966 (+28%) | 0.4049 (+82%) | 0.4288 (−109%) | 0.2631 (−11%) | 0.2622 (+28%) | 0.2652 (−182%) | **0.5615 (+35%)** | **0.6384 (+47%)** | 0.3883 (+3%) |
| Gaussian | 0.3171 (+36%) | 0.4053 (+82%) | 0.4511 (−11%) | 0.2848 (+13%) | 0.2874 (+46%) | 0.2903 (−108%) | 0.4888 (+24%) | 0.5501 (+34%) | 0.5240 (+25%) |
| Blur+TVM | **0.3448 (+47%)** | 0.4025 (+82%) | 0.4093 (−195%) | 0.2745 (+2%) | 0.2723 (+35%) | 0.2676 (−175%) | 0.4677 (+21%) | 0.4898 (+26%) | 0.4655 (+15%) |
| EnsBTC | 0.3364 (+44%) | **0.4128 (+84%)** | 0.4563 (+12%) | 0.3262 (+61%) | 0.3243 (+73%) | **0.3324 (+17%)** | 0.4748 (+22%) | 0.5813 (+39%) | 0.5274 (+25%) |
| EnsJMG | 0.3098 (+33%) | 0.4045 (+82%) | 0.4574 (+16%) | **0.3281 (+63%)** | **0.3245 (+74%)** | 0.3290 (+7%) | 0.4901 (+24%) | 0.5968 (+41%) | 0.5363 (+27%) |
| EnsJMT | 0.3058 (+32%) | 0.4114 (+84%) | **0.4599 (+27%)** | 0.3261 (+61%) | 0.3228 (+72%) | 0.3307 (+12%) | 0.5288 (+30%) | 0.6372 (+47%) | 0.4160 (+8%) |
<!-- /table:p3_v2_full -->

- On the patch columns the attack removed very little (0.023 mAP on YOLO-World, 0.034 on Florence-2), so large negative percentages there are small absolute changes.
- The N=1,000 clean baselines (**YOLO-World 0.4765**, **Florence-2 0.3605**) belong to this subset. They are *not* the survey's N=5,000 clean baselines (Florence-2 **0.3300**, YOLO-World **0.4519**). The OCR clean value is 1.0 **by construction**.

> **Correction (kept):** earlier notes quoted YOLO PGD as `clean 0.4972 → 0.1425 → 0.4801` and a
> "stronger PGD" `0.4765 → 0.0614 → 0.4263`. Those match **no committed v2 result**; they trace to a
> 100-image scratch run. Cite only committed `summary*.json` values.

### Phase 3 v2 with 95% bootstrap intervals (N=1,000)

Phase 4 re-scored the committed v2 detections (same images, no new attack) and added paired image-level bootstrap intervals (1,000 resamples, seed 42). These are the paper's interval tables (`tab:ci-yolo`, `tab:ci-florence`).
- *+ defense* columns: the actual mAP with the transform, clean or under that attack.
- Clean cost: clean mAP with the transform minus without. Recovery: attacked mAP with the transform minus without.
- **Bold** marks an interval that excludes zero.

YOLO-World:

<!-- table:p3_ci_yolo -->
| Defense | Clean + defense | Clean cost [95% CI] | FGSM + defense | Recovery [95% CI] | PGD + defense | Recovery [95% CI] | Patch + defense | Recovery [95% CI] |
|---|---|---|---|---|---|---|---|---|
| **No defense** | 0.476 |  | 0.226 |  | 0.074 |  | 0.454 |  |
| JPEG | 0.452 | **−0.024 [−0.033, −0.015]** | 0.270 | **+0.044 [+0.037, +0.052]** | 0.364 | **+0.291 [+0.273, +0.315]** | 0.439 | **−0.015 [−0.022, −0.006]** |
| Median | 0.458 | **−0.018 [−0.026, −0.010]** | 0.293 | **+0.067 [+0.060, +0.076]** | 0.364 | **+0.290 [+0.279, +0.315]** | 0.450 | −0.004 [−0.012, +0.009] |
| TVM | 0.447 | **−0.029 [−0.038, −0.021]** | 0.297 | **+0.071 [+0.061, +0.082]** | 0.405 | **+0.331 [+0.317, +0.357]** | 0.429 | **−0.025 [−0.034, −0.016]** |
| Gaussian | 0.461 | **−0.015 [−0.024, −0.007]** | 0.317 | **+0.091 [+0.082, +0.103]** | 0.405 | **+0.332 [+0.318, +0.357]** | 0.451 | −0.003 [−0.010, +0.008] |
| Blur+TVM | 0.415 | **−0.062 [−0.073, −0.050]** | 0.345 | **+0.119 [+0.105, +0.133]** | 0.403 | **+0.329 [+0.314, +0.355]** | 0.409 | **−0.044 [−0.056, −0.034]** |
| EnsBTC | 0.465 | **−0.012 [−0.020, −0.004]** | 0.336 | **+0.110 [+0.100, +0.123]** | 0.413 | **+0.339 [+0.326, +0.366]** | 0.456 | +0.003 [−0.004, +0.012] |
| EnsJMG | 0.466 | **−0.011 [−0.018, −0.002]** | 0.310 | **+0.084 [+0.076, +0.095]** | 0.405 | **+0.331 [+0.318, +0.358]** | 0.457 | +0.004 [−0.004, +0.016] |
| EnsJMT | 0.467 | **−0.009 [−0.015, −0.002]** | 0.306 | **+0.080 [+0.072, +0.092]** | 0.411 | **+0.338 [+0.324, +0.364]** | 0.460 | +0.006 [−0.001, +0.015] |
<!-- /table:p3_ci_yolo -->

Florence-2 detection:

<!-- table:p3_ci_florence -->
| Defense | Clean + defense | Clean cost [95% CI] | FGSM + defense | Recovery [95% CI] | PGD + defense | Recovery [95% CI] | Patch + defense | Recovery [95% CI] |
|---|---|---|---|---|---|---|---|---|
| **No defense** | 0.361 |  | 0.273 |  | 0.224 |  | 0.327 |  |
| JPEG | 0.352 | −0.008 [−0.021, +0.004] | 0.320 | **+0.047 [+0.036, +0.059]** | 0.316 | **+0.092 [+0.080, +0.111]** | 0.327 | +0.000 [−0.009, +0.013] |
| Median | 0.353 | −0.008 [−0.021, +0.007] | 0.319 | **+0.046 [+0.035, +0.061]** | 0.320 | **+0.096 [+0.081, +0.112]** | 0.329 | +0.002 [−0.006, +0.015] |
| TVM | 0.289 | **−0.071 [−0.088, −0.060]** | 0.263 | −0.010 [−0.028, +0.005] | 0.262 | **+0.038 [+0.020, +0.053]** | 0.265 | **−0.062 [−0.077, −0.050]** |
| Gaussian | 0.291 | **−0.069 [−0.085, −0.056]** | 0.285 | +0.012 [−0.003, +0.027] | 0.287 | **+0.063 [+0.048, +0.080]** | 0.290 | **−0.036 [−0.050, −0.025]** |
| Blur+TVM | 0.265 | **−0.095 [−0.115, −0.081]** | 0.274 | +0.001 [−0.016, +0.016] | 0.272 | **+0.048 [+0.029, +0.066]** | 0.268 | **−0.059 [−0.076, −0.043]** |
| EnsBTC | 0.352 | −0.009 [−0.020, +0.008] | 0.326 | **+0.053 [+0.042, +0.070]** | 0.324 | **+0.100 [+0.087, +0.120]** | 0.332 | +0.006 [−0.004, +0.024] |
| EnsJMG | 0.351 | −0.009 [−0.020, +0.008] | 0.328 | **+0.055 [+0.043, +0.072]** | 0.324 | **+0.100 [+0.087, +0.120]** | 0.329 | +0.002 [−0.004, +0.017] |
| EnsJMT | 0.356 | −0.004 [−0.015, +0.011] | 0.326 | **+0.053 [+0.043, +0.069]** | 0.323 | **+0.099 [+0.084, +0.119]** | 0.331 | +0.004 [−0.004, +0.018] |
<!-- /table:p3_ci_florence -->

Source: `results/phase4/rescore_phase3_v2/summary_rescore_phase3_v2_latest.json`. The paper's `tab_ci_*.tex` are written from it by `docs/paper_draft/make_tables.py`.

### Phase 3 — Survey, N=5,000

The Florence-2 detection rows are read from committed files produced by `archive/Phase3/run_survey_florence_detection.py --tier survey --num-images 5000`. That run has 19 defense rows (12 single transforms and 7 ensembles), `novel: false` and 0 failed images.

<!-- table:p3_survey_best -->
| Track | Clean | Attacked | Best defense | Defended | Recovery | Best single transform | Source file |
|---|---|---|---|---|---|---|---|
| Florence-2 det. · FGSM | 0.3300 | 0.2488 | EnsJBT | **0.2942** | +55.9% | Bilateral 0.2938 | `results/results_survey_florence_detection_fgsm/summary_fgsm.json` |
| Florence-2 det. · PGD | 0.3300 | 0.2005 | EnsJMT | **0.2961** | +73.8% | Bilateral 0.2956 | `results/results_survey_florence_detection_pgd/summary_pgd.json` |
| Florence-2 det. · Patch | 0.3300 | 0.2958 | Bilateral | **0.3165** | +60.6% | Bilateral 0.3165 | `results/results_survey_florence_detection_patch/summary_patch.json` |
<!-- /table:p3_survey_best -->

The YOLO-World and OCR rows below are from the earlier paper's tables (Table 3 and Table 5). The runs were real, but **their summary files no longer exist**. The YOLO-World clean value is now also backed by the committed Phase-4 file.

| Track | Clean | FGSM | PGD | Patch | Source |
|---|---|---|---|---|---|
| YOLO-World detection (mAP) | 0.4519 | 0.2051 | 0.0619 | 0.4257 | earlier paper Table 3; clean reproduced in `results/phase4/clean_control_yolo/` |
| Florence-2 OCR (self-consistency) | 1.000 | 0.346 | 0.294 | 0.383 | earlier paper Table 5 |

- Bilateral (a survey addition, absent from the v2 set) is the strongest **single transform** on all three Florence-2 detection attacks at N=5,000, and wins Patch outright. The N=1,000 v2 ranking does not carry over.
- Random resize is the worst row on all three attacks (e.g. Patch 0.1449). This is a documented negative result, not a bug.
- The N=5,000 PGD alpha is eps/4 on all three tracks. Never mix in the N=1,000 OCR PGD row (alpha 0.003).

**Every defense at N=5,000** (the paper's appendix A). Rec. % = (defended − attacked) / (clean − attacked).
- In the patch columns the denominator is tiny (0.026 mAP on YOLO-World, 0.034 on Florence-2), so read the mAP there, not the percentage.
- † Random resize is in the committed Florence-2 files but left out of the paper's detection tables: its detection numbers are a box-misalignment artefact.

YOLO-World detection (mAP), from the earlier paper's Table 3 (`docs/paper_draft/tab_yolo_n5000.tex`):

<!-- table:p3_n5000_yolo -->
|  | FGSM mAP | Rec. % | PGD mAP | Rec. % | Patch mAP | Rec. % |
|---|---|---|---|---|---|---|
| No attack, no defense | 0.4519 |  | 0.4519 |  | 0.4519 |  |
| Attacked, no defense | 0.2051 |  | 0.0619 |  | 0.4257 |  |
| *Single transforms* |  |  |  |  |  |  |
| Gaussian (σ=1) | 0.2948 | 36.3 | 0.3871 | 83.4 | 0.4261 | 1.4 |
| TVM (w=0.05) | 0.2757 | 28.6 | 0.3847 | 82.8 | 0.4017 | −92.1 |
| Blur+TVM | 0.3265 | 49.2 | 0.3815 | 81.9 | 0.3840 | −159.6 |
| Median (3×3) | 0.2750 | 28.3 | 0.3525 | 74.5 | 0.4299 | 16.1 |
| JPEG (q=75) | 0.2503 | 18.3 | 0.3377 | 70.7 | 0.4133 | −47.4 |
| Bilateral | 0.2268 | 8.8 | 0.3024 | 61.7 | 0.4206 | −19.6 |
| Anisotropic | 0.2151 | 4.1 | 0.3006 | 61.2 | 0.4085 | −65.9 |
| NLM (h=0.8σ̂) | 0.2018 | −1.3 | 0.1793 | 30.1 | 0.4225 | −12.3 |
| Ordered dithering | 0.2191 | 5.7 | 0.1581 | 24.7 | 0.4139 | −45.2 |
| Bit-depth (4-bit) | 0.2189 | 5.6 | 0.1382 | 19.6 | 0.4150 | −41.0 |
| SVD (top 90% of rank) | 0.2055 | 0.2 | 0.0638 | 0.5 | 0.4259 | 0.8 |
| *Output-level ensembles* |  |  |  |  |  |  |
| Ens4W | 0.2896 | 34.2 | 0.3969 | 85.9 | 0.4358 | 38.5 |
| EnsBTC | 0.3134 | 43.9 | 0.3931 | 84.9 | 0.4317 | 22.6 |
| EnsJMT | 0.2832 | 31.6 | 0.3902 | 84.2 | 0.4347 | 34.4 |
| EnsBMT | 0.2739 | 27.9 | 0.3857 | 83.0 | 0.4347 | 34.3 |
| EnsJMG | 0.2882 | 33.7 | 0.3853 | 82.9 | 0.4366 | 41.5 |
| EnsJBT | 0.2664 | 24.8 | 0.3834 | 82.4 | 0.4298 | 15.6 |
| EnsJTS | 0.2617 | 22.9 | 0.3751 | 80.3 | 0.4333 | 28.8 |
<!-- /table:p3_n5000_yolo -->

Florence-2 detection (mAP, geometric pseudo-scores), from the committed `results/results_survey_florence_detection_{fgsm,pgd,patch}/summary_*.json`:

<!-- table:p3_n5000_florence -->
|  | FGSM mAP | Rec. % | PGD mAP | Rec. % | Patch mAP | Rec. % |
|---|---|---|---|---|---|---|
| No attack, no defense | 0.3300 |  | 0.3300 |  | 0.3300 |  |
| Attacked, no defense | 0.2488 |  | 0.2005 |  | 0.2958 |  |
| *Single transforms* |  |  |  |  |  |  |
| Bilateral | 0.2938 | 55.4 | 0.2956 | 73.5 | 0.3165 | 60.6 |
| JPEG | 0.2922 | 53.5 | 0.2937 | 72.0 | 0.3044 | 25.1 |
| Median | 0.2910 | 52.0 | 0.2881 | 67.7 | 0.3032 | 21.5 |
| Anisotropic diffusion | 0.2855 | 45.2 | 0.2856 | 65.7 | 0.3028 | 20.4 |
| Gaussian | 0.2621 | 16.3 | 0.2625 | 47.9 | 0.2719 | −70.2 |
| Dithering | 0.2610 | 15.0 | 0.2529 | 40.5 | 0.2852 | −31.2 |
| Blur+TVM | 0.2467 | −2.6 | 0.2457 | 34.9 | 0.2437 | −152.7 |
| TVM | 0.2440 | −5.9 | 0.2410 | 31.3 | 0.2494 | −136.3 |
| Bit depth | 0.2321 | −20.7 | 0.2191 | 14.3 | 0.2477 | −141.2 |
| NLM | 0.2420 | −8.4 | 0.2058 | 4.1 | 0.2791 | −49.0 |
| SVD | 0.2462 | −3.2 | 0.1990 | −1.2 | 0.2920 | −11.3 |
| Random resize † | 0.1389 | −135.5 | 0.1329 | −52.2 | 0.1449 | −442.5 |
| *Ensembles* |  |  |  |  |  |  |
| EnsJMT | 0.2931 | 54.6 | 0.2961 | 73.8 | 0.3051 | 27.0 |
| EnsJMG | 0.2934 | 54.9 | 0.2957 | 73.6 | 0.3057 | 29.0 |
| EnsBTC | 0.2915 | 52.6 | 0.2955 | 73.3 | 0.3014 | 16.3 |
| EnsJBT | 0.2942 | 55.9 | 0.2953 | 73.3 | 0.3105 | 42.9 |
| EnsBMT | 0.2931 | 54.5 | 0.2938 | 72.1 | 0.3099 | 41.2 |
| Ens4W | 0.2897 | 50.4 | 0.2927 | 71.2 | 0.3016 | 17.0 |
| EnsJTS | 0.2792 | 37.5 | 0.2745 | 57.2 | 0.2994 | 10.3 |
<!-- /table:p3_n5000_florence -->

Florence-2 OCR (self-consistency), from the earlier paper's Table 5 (`docs/paper_draft/tab_ocr_n5000.tex`):

<!-- table:p3_n5000_ocr -->
|  | FGSM | PGD | Patch |
|---|---|---|---|
| No attack, no defense | 1.000 | 1.000 | 1.000 |
| Attacked, no defense | 0.346 | 0.294 | 0.383 |
| *Single transforms* |  |  |  |
| TVM (w=0.05) | 0.598 | 0.654 | 0.406 |
| Bilateral | 0.482 | 0.637 | 0.380 |
| Anisotropic | 0.507 | 0.630 | 0.372 |
| JPEG (q=75) | 0.522 | 0.616 | 0.407 |
| Gaussian (σ=1) | 0.535 | 0.609 | 0.572 |
| BM3D | 0.569 | 0.598 | 0.360 |
| Random resize-and-pad | 0.532 | 0.584 | 0.567 |
| Median (3×3) | 0.452 | 0.546 | 0.581 |
| Bit-depth (4-bit) | 0.445 | 0.521 | 0.349 |
| Ordered dithering | 0.437 | 0.516 | 0.343 |
| Blur+TVM | 0.470 | 0.487 | 0.477 |
| NLM (h=0.8σ̂) | 0.367 | 0.383 | 0.367 |
| SVD (top 90% of rank) | 0.350 | 0.309 | 0.385 |
| *Output-level ensembles (consensus vote)* |  |  |  |
| EnsJBT | 0.537 | 0.661 | 0.401 |
| Ens4W | 0.561 | 0.652 | 0.566 |
| EnsBMT | 0.530 | 0.651 | 0.427 |
| EnsJMT | 0.555 | 0.649 | 0.442 |
| EnsJMG | 0.522 | 0.620 | 0.578 |
| EnsBTC | 0.502 | 0.586 | 0.550 |
| EnsJTS | 0.507 | 0.580 | 0.405 |
<!-- /table:p3_n5000_ocr -->

Interpretation rule used in this repo: **prioritize recovery from the attacked score over naive comparison with the clean baseline.**

## Repository Structure

```
.
├── phase3_common.py            # shared core: defenses, ensembles, NMS merge, eval, checkpoint.
│                               # STAYS AT THE ROOT — phase4/ imports it, and importing the SAME
│                               # module is what makes Phase-4 numbers comparable to Phase 3.
├── phase4/
│   ├── p4_core.py              # everything not model-specific: CLI, image selection, defense
│   │                           # branches, checkpointed loop, COCO/OCR scoring, reports,
│   │                           # the clean_control experiment
│   ├── p4_yolo.py              # ENTRY POINT — YOLOv8x-worldv2 (+ the BPDA/EOT adaptive attack)
│   ├── p4_florence.py          # ENTRY POINT — Florence-2 detection + OCR from one model load,
│   │                           # and the adaptive attack on <OD>
│   ├── p4_smolvlm.py           # ENTRY POINT — SmolVLM-256M yes/no presence questions
│   ├── p4_owl.py               # ENTRY POINT — OWL-ViT (fourth model): oblivious survey + adaptive
│   └── p4_logging.py           # run-artifact policy (run.log, ephemeral eval dumps)
├── scripts/
│   ├── setup_server.sh         # fresh server: conda env, COCO, weights, preflight, tests
│   ├── p4_launch.sh            # starts the Phase-4 plan in a detached tmux session
│   ├── p4_queue.sh             # runs one queue of commands with logs, done markers, one retry
│   ├── p4_status.sh · p4_health.py   # read-only progress and health checks
│   ├── p4_subset_score.py      # scores the first K images of a finished adaptive checkpoint
│   ├── p4_ledger.py            # writes docs/PHASE4_JOB_LEDGER.md from the local queue ledgers
│   ├── make_readme_tables.py   # writes every README results table from the summaries (--check)
│   └── p4_boot_survey.py       # paired bootstrap CIs for an oblivious_survey run (ensemble vs members)
├── tests/                      # Phase-4 checks; none need a GPU, weights or COCO
├── results/
│   ├── phase4/<experiment>_<model>/   # Phase-4 runs (timestamped summaries + run logs)
│   └── results_*/              # Phase-3 outputs (summary*.json, run.log, .png tracked)
├── docs/
│   ├── paper_draft/            # main.tex, refs.bib, table/figure generators, Overleaf zip
│   ├── PHASE4_RESULTS_SUMMARY.md · PHASE4_JOB_LEDGER.md · memory.md · HANDOFF_A5000.md
│   └── CV_Research_Paper___CVPR_Format.pdf · Presentation.pdf   # earlier report
├── figures/                    # Final_Results_Images/, Result_Images/
├── README.md · CLAUDE.md · environment.yml · .gitignore
└── archive/                    # read-only history (Phase3/, Scripts_Extra/, Logs_Extra/, Very_OLD/)
```

The data and model assets the runs use are gitignored and fetched separately:
- `val2017/` and `annotations/instances_val2017.json`;
- `yolov8x-worldv2.pt`;
- `weights/`.

## Environment Setup (fresh server)

```bash
git clone <repo> && cd Adversarial_Attack_Defence_Paper
bash scripts/setup_server.sh     # Miniforge if no conda, env, COCO val2017, all weights, preflight, tests
```

**What the script does:**
- It is re-runnable: each step skips what already exists.
- It needs internet, so on a cluster run it on the login node.
- Everything lands inside the repo, all gitignored: `val2017/`, `annotations/`, `yolov8x-worldv2.pt`, `weights/clip/` and the Florence-2 / SmolVLM cache in `weights/hf/`.
- torch 2.6.0+cu118 covers sm_50 to sm_90. The script warns on Blackwell GPUs, which need a cu128 torch.

In every new shell, tmux pane or SLURM job:

```bash
conda activate vlm_ftune && cd <repo> && export HF_HOME=$PWD/weights/hf
export HF_HUB_OFFLINE=1          # only on GPU nodes without internet
```

## Dataset Layout

```text
./val2017/
./annotations/instances_val2017.json
```

- The Phase-4 entry points find both paths on their own: they try `./val2017`, `./Dataset/val2017` and a few others. Pass `--image-dir` / `--ann-file` if your layout differs.
- The archived Phase-3 scripts have hardcoded defaults that differ by track; see `archive/Phase3/README.md`.

## Phase 4 — how to run

**Rules:**
- One process = one model = one GPU = one tmux pane.
- `--preflight` needs no GPU and lists anything missing.
- `--smoke-test` runs 2 images into a gitignored `smoke_test/` subdirectory.
- **A real run stops with an error if the image directory holds fewer than `--n` images.**

### The whole plan in one command

```bash
bash scripts/p4_launch.sh test             # every job of the plan on 5 images
bash scripts/p4_launch.sh full             # the real runs (the commands below)
bash scripts/p4_launch.sh full --dry-run   # write the queues, start nothing
bash scripts/p4_status.sh full             # one-screen progress
python3 scripts/p4_health.py full          # verdict per queue: OK / WAITING / STALLED? / DEAD? / FINISHED
tmux attach -t p4_full                     # Ctrl-b d detaches
```

**How the launcher works:**
- It writes one queue file per tmux window to `logs/phase4_queue/<mode>/queues/`.
- Each job's output goes to `<queue>.console.log` and `status.tsv`.
- After a reboot, run the same launch command again. Finished jobs are skipped, and unfinished ones resume from their checkpoints.

### The exact commands behind the Phase-4 results

```bash
D="--deterministic --gpu 0"
A="--attack-space native --pgd-select best --pgd-iters 50 --alpha 0.002"

# Adaptive vs oblivious, YOLO-World, 1,000 images (paper table tab:adaptive)
for t in ens_jpeg_median_gaussian median; do for s in 1 2 3; do
  python phase4/p4_yolo.py --experiment adaptive_bpda $A $D --n 1000 --target $t --seed $s \
      --log-name run_${t}_s$s.log
done; done
python phase4/p4_yolo.py --experiment adaptive_bpda $A $D --n 1000 --target ens_4way --seed 1 \
    --log-name run_ens_4way_s1.log

# Checks: mean objective, identity median gradient, 100 steps, eps 0.1
python phase4/p4_yolo.py --experiment adaptive_bpda $A $D --n 1000 --target ens_jpeg_median_gaussian \
    --seed 1 --eot-loss mean --log-name run_mean.log            # same for --target ens_4way
python phase4/p4_yolo.py --experiment adaptive_bpda $A $D --n 1000 --target median --seed 1 \
    --median-surrogate identity --log-name run_identity.log
python phase4/p4_yolo.py --experiment adaptive_bpda $D --attack-space native --pgd-select best \
    --n 300 --target ens_jpeg_median_gaussian --seed 1 --pgd-iters 100 --alpha 0.002 \
    --output-dir results/phase4/adaptive_bpda_yolo_iters100_n300     # and --pgd-iters 50 -> _iters50_n300
python phase4/p4_yolo.py --experiment adaptive_bpda $D --attack-space native --pgd-select best \
    --n 300 --target ens_jpeg_median_gaussian --seed 1 --pgd-iters 50 --eps 0.1 --alpha 0.0067 \
    --output-dir results/phase4/adaptive_bpda_yolo_eps0.1

# Adaptive vs oblivious, Florence-2 detection, 1,000 images (always pixel space)
for t in ens_jpeg_median_gaussian median; do
  python phase4/p4_florence.py --experiment adaptive_bpda --pgd-select best --pgd-iters 50 \
      --alpha 0.002 $D --n 1000 --target $t --seed 1 --log-name run_${t}_s1.log
done

# Clean cost, every transform, 5,000 images
python phase4/p4_yolo.py --experiment clean_control --all-transforms $D --n 5000 --seed 1
python phase4/p4_florence.py --experiment clean_control --all-transforms $D --tracks florence_det \
    --n 5000 --seed 1 --output-dir results/phase4/clean_control_florence_det
python phase4/p4_florence.py --experiment clean_control --all-transforms $D --tracks florence_ocr \
    --n 5000 --seed 1 --output-dir results/phase4/clean_control_florence_ocr

# SmolVLM-256M, 1,000 images
python phase4/p4_smolvlm.py $D --n 1000 --seed 1

# OWL-ViT (fourth model): every transform under oblivious PGD, then the adaptive attack
O="--pgd-iters 50 --alpha 0.002 --pgd-select best --seed 1"
python phase4/p4_owl.py --experiment oblivious_survey --n 1000 $O $D
python phase4/p4_owl.py --experiment adaptive_bpda --n 1000 $O $D --target median --log-name run_median_s1.log
python phase4/p4_owl.py --experiment adaptive_bpda --n 1000 $O $D --target ens_jpeg_median_gaussian \
    --log-name run_ens_jpeg_median_gaussian_s1.log
python phase4/p4_owl.py --experiment adaptive_bpda --n 300 $O $D --target ens_jpeg_median_gaussian \
    --eot-loss mean --output-dir results/phase4/adaptive_bpda_owl_n300 --log-name run_mean.log
python scripts/p4_boot_survey.py results/phase4/oblivious_survey_owl/checkpoint_survey_*.pkl \
    results/phase4/oblivious_survey_owl/summary_oblivious_survey_owl_seed1_latest.json \
    results/phase4/oblivious_survey_owl/summary_bootstrap_oblivious_survey_owl_seed1.json --B 1000

# Smaller budget, 300 images: eps 0.016 (4/255), alpha 0.00107 (eps/15); also --eps 0.03 for JPEG/Gaussian
for t in median jpeg gaussian ens_jpeg_median_gaussian; do
  python phase4/p4_yolo.py --experiment adaptive_bpda --attack-space native --pgd-select best $D \
      --n 300 --seed 1 --pgd-iters 50 --eps 0.016 --alpha 0.00107 --target $t \
      --output-dir results/phase4/adaptive_bpda_yolo_eps4of255_n300 --log-name run_$t.log
done      # plus --eot-loss mean for EnsJMG; Florence: p4_florence.py, Median and EnsJMG

# Checks that need no GPU
python phase4/p4_yolo.py --preflight
python tests/test_offline_logic.py      # 39 torch-free checks
python tests/test_flow_stubbed.py       # 21 driver checks, model + attacks stubbed
CUDA_VISIBLE_DEVICES='' python tests/test_attack_units.py   # 23 CPU-torch attack checks
```

**Shared flags:**
- Run control: `--n`, `--select {head,random}`, `--seed`, `--tracks`, `--gpu`, `--device`, `--checkpoint-every`, `--no-checkpoint`, `--keep-eval-dumps`, `--no-per-image`, `--log-every N`, `--verbose-eval`, `--log-name`.
- Execution flags: `--deterministic` and `--eot-batch {auto,on,off}`. These are recorded in the summary but stay out of the checkpoint signature and the file name.
- Defense hyperparameters. The defaults match Phase 3: JPEG q75, median 3×3, TVM w=0.05/200 iters, sigma=1.0, SVD 90%, NMS IoU 0.5.

**How the seed works:**
- `--select head` (the default) takes the first `--n` sorted images, the same subset the Phase-3 runs scored.
- `--seed` then changes only the PGD start point and the SmolVLM questions, never the images.
- Every summary records the image ids and their SHA1 digest.

### Measured throughput (RTX 5000 Ada 32 GB, Phase-4 runs, 17–18 Nov 2026)

These are seconds per image as the runs reported them. Up to 8 jobs shared the one GPU in round 1 and up to 9 in round 2, so a job running alone is faster. The highest GPU memory logged (round 1, 8 jobs, local health log) was 30,517 of 32,760 MiB.

| Run | s/image | images |
|---|---|---|
| YOLO `clean_control --all-transforms` | 1.8 | 5,000 |
| YOLO adaptive, Median | 10–12 | 1,000 |
| YOLO adaptive, EnsJMG | 15–17 | 1,000 |
| YOLO adaptive, Ens4W (max / mean objective) | 26.5 / 13.7 | 1,000 |
| Florence adaptive, Median / EnsJMG | 45.5 / 67.6 | 1,000 |
| Florence `clean_control`, detection / OCR | 16.9 / 14.8 | 5,000 |
| SmolVLM presence QA | 26.7 | 1,000 |
| OWL-ViT `oblivious_survey` (11 transforms + 7 ensembles) | 11.8 | 1,000 |
| OWL-ViT adaptive, Median / EnsJMG | 14.0 / 18.4 | 1,000 |
| OWL-ViT adaptive, EnsJMG mean objective | 29.5 | 300 |
| YOLO adaptive, 300-image runs (4/255, 7/255, 100 steps, eps 0.1) | 13.7–28.5 | 300 |
| Florence adaptive at 4/255, Median / EnsJMG | 58.8 / 77.4 | 300 |
| Florence adaptive, EnsJMG mean objective | 54.3 | 300 |

`--deterministic` cost about 19% on YOLO in a 10-image test before the runs (7.2 against 6.1 s/image; no file records it). It makes YOLO runs bit-identical, but not Florence-2 or OWL-ViT runs.

### The experiments

| Experiment | Model(s) | What it answers |
|---|---|---|
| `clean_control` | YOLO, Florence | Do these transforms counter *attacks*, or do they just help or hurt images in general? Scores every transform on clean images. `--all-transforms` adds bit depth, dithering, NLM and anisotropic diffusion (OCR also BM3D and random resize) under its own `_alltransforms` summary name. |
| `adaptive_bpda` | YOLO, Florence `<OD>` | What happens when the attacker knows the defense? PGD runs through the `--target` defense with BPDA and EOT, and is compared with the same PGD run obliviously from the **same seeded start**. Reports clean / oblivious / adaptive mAP with and without the defense. |
| `presence_qa` (`p4_smolvlm.py`) | SmolVLM-256M | Do the transforms protect a chat-style VLM? Yes/no questions built from the COCO annotations, oblivious PGD only. |
| `oblivious_survey` (`p4_owl.py`) | OWL-ViT | Clean cost and oblivious-PGD recovery of every transform and ensemble in one run (the Phase-3 style table). OWL-ViT also runs `adaptive_bpda`. |

Adaptive-attack flags. Each enters the checkpoint signature, the summary's `attack_flags` block and the summary name:

| Flag | Default | Meaning |
|---|---|---|
| `--target` | `ens_jpeg_median_gaussian` (EnsJMG) | Also `ens_4way` (Ens4W), `ens_jpeg_median_tvm` (EnsJMT), `ens_blur_tvm_combo` (EnsBTC), and the solos `median`, `jpeg`, `gaussian`, `tvm`, `blur_tvm`. Members are read from `phase3_common.TIER1_ENSEMBLES`. |
| `--eot-loss` | `max` | `max`: per anchor, suppress the most confident branch (Florence: raise the least-damaged branch's loss). `mean`: the mean of the branch objectives. **On YOLO `mean` was the stronger attack, so report the worst case of both.** |
| `--restarts` | `1` | Seeded restarts per image; keeps the lowest surrogate objective. |
| `--median-surrogate` | `true` | `true`: exact almost-everywhere gradient of the median. `identity`: BPDA straight-through. JPEG and TVM always use identity; Gaussian always uses its true gradient. |
| `--quantize` | `truncate` | Final uint8 conversion. |
| `--pgd-select` | `last` | `best` keeps whichever of the last iterate and the best-seen iterate has the lower objective after quantization. Used for every Phase-4 run. |
| `--attack-space` | `letterbox` | `native` optimizes at native resolution, so the defense runs at the deployed scale. Used for every Phase-4 YOLO run. Florence always attacks in native pixel space. |

**Summary file names:**
- Pattern: `summary_adaptive_bpda_<model>_<target>_seed<S>[_<flag>-<value>...]_n<N>_<timestamp>.json`, plus a `_latest` copy.
- Every target, seed and ablation therefore keeps its own citable `_latest` in one resumable run directory.
- `--eps` and `--pgd-iters` diagnostics need their own `--output-dir`.

**Per-pixel budget:**
- In `native` space the uint8 attack output is clipped to the clean image ± floor(eps×255)/255. At eps 0.03 that is 7/255, since 0.03×255 = 7.65.
- The scored defense is always `phase3_common`'s. The BPDA surrogates only shape the attacker's gradient.

### Phase 3 (archived)

The nine locked v2 scripts/notebooks and the three N=5,000 survey scripts live in `archive/Phase3/`. **Read `archive/Phase3/README.md` before touching them:**
- they need `PYTHONPATH=.` from the repo root;
- they write to old root paths;
- their dataset paths differ by track;
- their PGD alpha is not uniform (see below).

> ⚠️ **PGD step size is NOT uniform across tracks.** `archive/Phase3/PGD_Florence2_OCR_Robust.py:642`
> defaults to `--alpha 0.003` (= eps/10). That is the value behind the committed N=1,000 OCR PGD
> number (`results/results_pgd_florence2_ocr_robust/summary.json` records `"alpha": 0.003`). Every
> other Phase-3 PGD track uses **alpha = eps/4 = 0.0075**. Do not "fix" the 0.003; the committed
> result depends on it.

> ⚠️ **eps=0.03 is not the same budget for both models in Phase 3.** Florence added it in normalized
> (ImageNet) space, about 1.7/255 in pixels; YOLO added it in raw [0,1], 7.65/255. Phase 4 attacks both
> in pixel space at 7/255.

## Run Artifacts and Log Policy

**Every Phase-4 run produces three shareable files, and nothing large.** `phase4/p4_logging.py` enforces this:

| File | Size | Git | What it is |
|---|---|---|---|
| `summary_*.json` | KB | **committed** | the numbers, the config, the chosen image ids |
| `summary_*.md` | KB | **committed** | the readable table |
| `run.log` / `run_<job>.log` | tens of KB | **committed** | full console transcript: host, git commit, exact command, exit status |
| `per_image_*.json` | ~1–3 MB | ignored | per-image audit trail (`--no-per-image` skips it) |
| `checkpoint_*.pkl` | large | ignored | config-keyed resume state |
| per-condition COCO dumps | **hundreds of MB** | **never written** | sent to a temp dir and deleted (`--keep-eval-dumps` keeps them) |

**Log hygiene:**
- pycocotools output is replaced by one `[eval]` line per condition (`--verbose-eval` restores it).
- The heartbeat prints every 100 images.
- The full markdown table goes only to the `.md` file.

**Summaries are timestamped and never overwritten.** A `_latest.json`/`_latest.md` copy gives a stable file to cite. The run *directory* stays fixed, so a crashed run resumes from its checkpoint. Checkpoint names carry the config signature (`checkpoint_<track>_<sig8>.pkl`), so different configurations do not clash.

**`.gitignore` matches files, not directories:** `results/**/*.json` + `!results/**/summary*.json` + `!results/**/run.log`.
- Never write `results/*`. Excluding a directory makes every `!` negation inside it unrescuable and silently drops all committed summaries.
- Every file written into `results/` must start with `summary` to be committed.
- `results/**/smoke_test/` is ignored wholesale.

**Phase-3 scripts are different:** they still write per-condition dumps into their output directory. `.gitignore` keeps them out of git. Delete `results_*/[!s]*.json` by hand before sharing one of those directories.

## Citation

If you use this repository, cite the paper draft in `docs/paper_draft/` and the earlier report `docs/CV_Research_Paper___CVPR_Format.pdf`. Keep the model, data and attack citations used there: Florence-2, YOLO-World, SmolVLM, COCO, FGSM, PGD, the Patch attack, and BPDA/EOT (Athalye et al. 2018).
