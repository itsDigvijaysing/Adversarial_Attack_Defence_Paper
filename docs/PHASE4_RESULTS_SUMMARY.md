# Phase-4 Results Summary

As of 18 Nov 2026.

All Phase-4 runs finished on 18 Nov at 03:10: 20 GPU jobs and 0 failures.

**Main finding.** The defenses that looked strong in Phase 3 collapse once the attacker accounts for them. On YOLO-World the best ensemble keeps only 0.05–0.07 mAP (clean: 0.48). That is close to the undefended model under attack (0.02).

Every number below is taken from a `summary_*_latest.json` file under `results/phase4/` (listed at the end).

## 1. What we ran

| Experiment | Model | Images | Settings |
|---|---|---|---|
| Adaptive attack (BPDA+EOT) vs oblivious PGD | YOLOv8x-worldv2 | 1,000 | Median and EnsJMG with seeds 1, 2, 3; Ens4W with seed 1 |
| Adaptive attack (BPDA+EOT) vs oblivious PGD | Florence-2-base `<OD>` | 1,000 | Median and EnsJMG, seed 1 |
| Attack checks (steps, budget, objective, median gradient) | YOLO / Florence | 300–1,000 | see section 5 |
| Clean cost of every defense (no attack) | YOLO, Florence detection, Florence OCR | 5,000 | 11 single transforms + 7 ensembles (OCR adds BM3D and random resize) |
| Third model: yes/no object-presence questions | SmolVLM-256M-Instruct | 1,000 (2,000 questions) | Oblivious PGD only, 13 single transforms + 7 ensembles |

**Common settings:**
- Images: the first N COCO val2017 images in sorted order.
- Seed 1 unless stated.
- Budget eps = 0.03, which is at most 7/255 per pixel at native resolution.
- PGD: 50 steps, alpha = 0.002.
- `--deterministic` on.
- The defense code is the same `phase3_common.py` that Phase 3 used.

## 2. Main result: YOLO-World under the adaptive attack

All values are COCO mAP@[.5:.95] on 1,000 images.

- Clean, no defense: **0.476**.
- PGD, no defense: **0.015**. This is the 3-seed mean; seed 1 alone gives 0.018.

| Defense | Clean + defense | Oblivious PGD + defense | Adaptive, max objective | Adaptive, mean objective | Worst case |
|---|---|---|---|---|---|
| Median | 0.458 | 0.366 | 0.018 (3 seeds, sd 0.001) | same (one branch) | **0.017** |
| EnsJMG (JPEG + Median + Gaussian) | 0.466 | 0.407 | 0.073 (3 seeds, sd 0.002) | 0.048 | **0.048** |
| Ens4W (JPEG + Median + TVM + Gaussian) | 0.468 | 0.431 | 0.108 | 0.069 | **0.069** |

Column notes:
- **Oblivious PGD + defense:** the attacker ignores the defense. This is the Phase-3 threat model.
- **Adaptive:** the attacker differentiates through the defense, using BPDA for the parts with no gradient and EOT over the ensemble branches.
- **Worst case:** the lowest adaptive value for seed 1, taken over the two objectives.
- **Ens4W:** run with seed 1 only.

**What the table shows:**
- An oblivious attack leaves the defended model at 0.37–0.43 mAP.
- An adaptive attack with the same 7/255 budget brings it down to 0.017–0.069.
- Median ends at the undefended level, and the ensembles end just above it.

## 3. Florence-2 under the adaptive attack

All values are mAP on 1,000 images, seed 1.

- Clean, no defense: **0.360**.
- Oblivious PGD, no defense: **0.134–0.138**. Two identical runs give two values because Florence is not bit-repeatable.

| Defense | Clean + defense | Oblivious PGD + defense | Adaptive PGD + defense |
|---|---|---|---|
| Median | 0.350 | 0.256 | **0.148** |
| EnsJMG | 0.353 | 0.264 | **0.166** |

The pattern is the same as YOLO's, but the gap is smaller:
- Against an oblivious attack, a defense adds about +0.12 mAP.
- Against an adaptive attack, it adds only +0.01 to +0.03.

On EnsJMG (first 300 images), the mean and max objectives give the same result: 0.201 and 0.199.

## 4. How this differs from Phase 3 and the current draft

These are YOLO values on the same 1,000 images.

| | Phase 3 (the paper so far) | Phase 4, oblivious | Phase 4, adaptive (worst case) |
|---|---|---|---|
| Attack | PGD, 10 steps, alpha 0.0075, letterbox space | PGD, 50 steps, alpha 0.002, native resolution | BPDA+EOT through the defense, 50 steps |
| No defense | 0.074 | 0.015 | 0.015 |
| Median | 0.364 | 0.366 | 0.017 |
| EnsJMG | 0.405 | 0.407 | 0.048 |

**What is different:**
- **The defended numbers reproduce Phase 3.** Oblivious-defended is 0.366 against 0.364, and 0.407 against 0.405. The clean scores also match exactly: YOLO 0.4519 on 5,000 images, 0.4764 on 1,000 images, and Florence 0.3304 / 0.3604. So the code and data are consistent with the paper.
- **The new attack is stronger on the undefended model** (0.015 vs 0.074) because it runs 50 steps at native resolution. That makes the defenses look slightly *better* under the Phase-3 threat model.
- **The real change is the threat model.** Once the attacker knows the defense, the Phase-3 recovery (+0.29 to +0.33 mAP) disappears. The ensembles we recommended in Phase 3 do not hold up.
- **This matches prior work.** He et al. (WOOT 2017) showed the same effect for ensembles of weak defenses on image classifiers. We show it on open-vocabulary detectors and a VLM.
- **Florence Phase-3 numbers are not comparable.** Phase 3 attacked Florence in normalized space, about 1.7/255 per pixel. Phase 4 attacks it in pixel space at 7/255.

## 5. Checks that the attack is sound

| Check | Setting | Result | What it tells us |
|---|---|---|---|
| Adaptive vs oblivious objective | all 1,000-image runs | adaptive objective lower on 999–1,000 of 1,000 images | the optimizer does what it should |
| More steps | EnsJMG, 300 images | 50 steps 0.086 → 100 steps 0.054 | not converged at 50 steps; our numbers are an upper bound |
| Larger budget | EnsJMG, eps 0.1, 300 images | 0.023 | mAP keeps falling as the budget grows; no sign of gradient masking |
| Max vs mean objective | YOLO EnsJMG and Ens4W; Florence EnsJMG | YOLO: mean is stronger (0.071 → 0.048, 0.108 → 0.069). Florence: equal | report the worst case over both |
| Median gradient | true vs identity (BPDA) | 0.017 vs 0.042 | the true median gradient is the stronger attack |
| Repeatability | `--deterministic`, same seed | YOLO bit-identical; Florence varies about 0.005 mAP | state repeatability per model |

## 6. Clean cost of every defense (5,000 images, no attack)

**Baselines with no defense:** YOLO 0.4519 mAP, Florence detection 0.3304 mAP, OCR 1.000.

The OCR value is self-consistency: how closely the defended OCR text matches the model's own clean output. It is not accuracy.

| Defense | YOLO mAP change | Florence det. mAP change | OCR self-consistency |
|---|---|---|---|
| EnsJTS | +0.000 | −0.007 | 0.843 |
| Ens4W | −0.004 | −0.009 | 0.746 |
| EnsJBT | −0.004 | −0.005 | 0.789 |
| EnsBMT | −0.004 | −0.005 | 0.788 |
| EnsJMG | −0.005 | −0.006 | 0.713 |
| EnsJMT | −0.005 | −0.006 | 0.746 |
| EnsBTC | −0.010 | −0.010 | 0.644 |
| SVD (keep 90%) | 0.000 | −0.002 | 0.953 |
| NLM | −0.002 | −0.020 | 0.886 |
| Bilateral | −0.006 | +0.001 | 0.821 |
| Bit depth | −0.007 | −0.050 | 0.797 |
| Dithering | −0.009 | −0.027 | 0.776 |
| Anisotropic diffusion | −0.012 | −0.014 | 0.798 |
| Median | −0.013 | −0.010 | 0.620 |
| Gaussian | −0.015 | −0.056 | 0.672 |
| TVM | −0.024 | −0.066 | 0.725 |
| JPEG | −0.026 | −0.011 | 0.673 |
| Blur+TVM | −0.057 | −0.081 | 0.507 |
| BM3D (OCR only) | | | 0.648 |
| Random resize (OCR only) | | | 0.707 |

**On detection, the ensembles are cheap.** They cost 0.010 mAP or less, while single transforms cost up to 0.081.

**On OCR, every transform is expensive.** Even on clean images, self-consistency drops to 0.51–0.95. That sets a ceiling on any OCR recovery.

## 7. Third model: SmolVLM-256M (yes/no object presence, 1,000 images)

- Clean accuracy: **0.856**.
- Under a 10-step oblivious PGD attack (eps 0.03): **0.439**.
- Questions are half yes and half no.

| Defense | Clean accuracy | Attacked accuracy | Share of damage recovered |
|---|---|---|---|
| Blur+TVM | 0.786 | **0.754** | 75% |
| Random resize | 0.825 | 0.748 | 74% |
| BM3D | 0.810 | 0.728 | 69% |
| TVM | 0.832 | 0.638 | 48% |
| Gaussian | 0.820 | 0.580 | 34% |
| EnsBTC | 0.831 | 0.538 | 24% |
| Ens4W | 0.835 | 0.532 | 22% |
| Median | 0.845 | 0.531 | 22% |
| EnsJMG | 0.834 | 0.523 | 20% |
| EnsJMT | 0.839 | 0.522 | 20% |
| EnsBMT | 0.843 | 0.507 | 16% |
| JPEG | 0.827 | 0.483 | 11% |
| EnsJBT | 0.837 | 0.479 | 10% |
| EnsJTS | 0.846 | 0.470 | 8% |
| Anisotropic diffusion | 0.846 | 0.465 | 6% |
| Bilateral | 0.842 | 0.453 | 3% |
| NLM, dithering, SVD, bit depth | 0.850–0.855 | 0.438–0.441 | 0% |

On SmolVLM the strongest smoothing transforms work best against an oblivious attack, and the ensembles recover only 8–24% of the damage. We did not run an adaptive attack on SmolVLM. Given sections 2–3, these recoveries should be read as upper bounds.

## 8. What it means for the paper, and caveats

**Message.** Input-transform defenses, including the ensembles, give open-vocabulary detectors almost no protection against an attacker who knows the defense. The large Phase-3 recoveries measure the oblivious threat model only.

**Text changes needed:**
- Present Table 5 as an upper bound, because more steps lower it further.
- Report the worst case over the objectives.
- Drop the claim that the max objective is the natural worst case, and the "union-aware" novelty claim. The mean objective was stronger on YOLO.
- State repeatability per model.
- Present the OCR clean cost as a ceiling on OCR recovery.

**Caveats to state in the paper:**
- The adaptive attacks and SmolVLM use 1,000 images (the first 1,000 sorted val2017 ids). The clean cost uses 5,000.
- Ens4W and both Florence targets ran with one seed. Median and EnsJMG on YOLO ran with three, and the seeds agree within 0.002 mAP.
- Florence is not bit-repeatable even with `--deterministic`: about 0.005 mAP between identical runs. YOLO is bit-identical.
- Florence detection scores are geometric, because Florence emits no confidences. Comparisons across conditions are fair, but absolute mAP is not a confidence-ranked mAP.
- The OCR metric is self-consistency, not accuracy.

## 9. Where the numbers come from

All folders are under `results/phase4/`. Each holds `summary_*_latest.json`, a matching `.md` file and the run logs.

| Folder | Contents |
|---|---|
| `adaptive_bpda_yolo/` | YOLO Median and EnsJMG (seeds 1–3), Ens4W, the mean-objective runs and the identity-gradient check |
| `adaptive_bpda_yolo_iters50_n300/`, `adaptive_bpda_yolo_iters100_n300/` | 50 vs 100 steps |
| `adaptive_bpda_yolo_eps0.1/` | eps 0.1 |
| `adaptive_bpda_florence/` | Florence Median and EnsJMG |
| `adaptive_bpda_florence_n300/` | Florence mean vs max objective |
| `clean_control_yolo/`, `clean_control_florence_det/`, `clean_control_florence_ocr/` | clean cost, 5,000 images |
| `presence_qa_smolvlm/` | SmolVLM |
