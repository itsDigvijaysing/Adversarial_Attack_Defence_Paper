# LLM Memory -- Florence-2 Adversarial Robustness Project

## Project State (updated 2026-04-11)
- Phase 1 DONE: FGSM + PGD on Florence-2-Base, COCO val2017. Paper submitted to APSCON IEEE.
- Phase 2 IN PROGRESS: 3 variant notebooks created, READY TO RUN after critical eval-loop bug fix.
- Old files in `OLD/` directory. Do NOT edit.

## CRITICAL BUG FOUND & FIXED (2026-04-11)

### Bug: Eval loop never executed FGSM attacks
`run_full_evaluation()` in cell 20 of ALL notebooks:
- Created condition keys like `fgsm_eps0.003`, `fgsm_eps0.01+tvm`, etc.
- Initialized them as empty lists `[]`
- **NEVER called `fgsm_attack()` or ran inference on attacked images**
- Only computed clean and clean+defense conditions
- Result: COCO eval got 0 detections for every attacked condition -> mAP=0.0000 everywhere
- All 3 variants (A, B, C) showed "NOT HELPFUL" because attacked mAP was artificial zero

### Fix applied to ALL 4 notebooks (A, B, C, and phase2_fgsm_variant2):
The eval loop now has 3 sections per image:
1. Clean baseline: `run_inference(pil_img)`
2. Clean + defenses: `run_inference(dfunc(pil_img))`
3. **FGSM attacked (was missing)**: for each epsilon:
   - `adv_img = fgsm_attack(pil_img, eps=eps)`
   - `run_inference(adv_img)` -- attacked, no defense
   - `run_inference(dfunc(adv_img))` -- attacked + each defense

### Previous bugs (all fixed in Phase 2 notebooks):
1. JPEG before attack (useless) -> now after attack
2. Scores all 1.0 -> heuristic scores: `0.6 + 0.2*area_ratio + 0.15*(1-center_dist)`
3. No label mapping -> `FLORENCE_TO_COCO` dict (~70 mappings) in cell 10
4. Missing `repetition_penalty=1.8, length_penalty=1.0` -> added to ALL generate() calls
5. Unfair baseline -> identical pipeline for all conditions
6. Clean mAP with defense never measured -> now measured

## File Map (current as of 2026-04-11)

### Active notebooks (Phase 2) -- ALL share same base code, differ only in defenses:
- **FGSM_Phase2_VariantA.ipynb** -- Classical Squeezing: JPEG(q=75), Gaussian Blur(s=1.0), Median Filter(3x3), Bit Depth(4-bit)
- **FGSM_Phase2_VariantB.ipynb** -- Advanced Denoising: TVM(w=0.05), NLM(h=6), SVD Spectral(90%), Random Resize+Pad
- **FGSM_Phase2_VariantC.ipynb** -- Combined Pipelines: JPEG->TVM->NLM, Blur->TVM, NLM->Resize, Full Pipeline(JPEG->TVM->NLM->Resize)
- **phase2_fgsm_variant2.ipynb** -- Original variant2, has TVM+NLM+JPEG+RandomResize+Combined (superset)
- Output dirs: `results_phase2_variantA/`, `results_phase2_variantB/`, `results_phase2_variantC/`

### Other files:
- phase2_fgsm_variant1.ipynb -- FGSM + 5 defenses (JPEG, blur, median, DiffPure, SVD) -- older
- plan.md, future.md -- planning docs
- environment.yml -- conda env spec for vlm_ftune
- OLD/ -- Phase 1 files (DO NOT EDIT)

## Notebook Structure (shared across A, B, C)
All 3 variants have identical structure, only cells 0 (title), 6 (config), 13-14 (defenses) differ:

| Cell | Content |
|------|---------|
| 0 | Title/overview markdown |
| 2 | Imports + GPU isolation (NUM_GPUS before torch import) |
| 4 | GPU diagnostics + device selection |
| 6 | Config: NUM_IMAGES, EPSILONS, defense flags, parameters, OUTPUT_DIR |
| 8 | Load model (Florence-2-Base) + dataset (COCO val2017) |
| 10 | FLORENCE_TO_COCO label mapping + _map_label() + _compute_score() + run_inference() + NMS |
| 12 | fgsm_attack() function |
| 14 | Defense functions + DEFENSES registry dict |
| 16 | Sanity check (verifies inference, attack, each defense) |
| 18 | evaluate_coco() helper using pycocotools COCOeval |
| 20 | **run_full_evaluation()** -- main loop (FIXED: now includes FGSM attack block) |
| 22 | COCO eval for all conditions |
| 24 | Results summary table |
| 26-27 | Visualization (bar charts + sample image comparison) |
| 29 | Net gain analysis |

## Key Technical Details
- Model: `microsoft/Florence-2-base` via HuggingFace transformers
- Attack: FGSM in NORMALIZED pixel space (ImageNet mean/std), loss = model(..., labels=target_ids).loss
- Target labels = model's own beam-search predictions on clean image (untargeted attack via label maximization)
- x_adv = x + eps * sign(grad), clamped to [-2.5, 2.5], denormalized to PIL
- Epsilons: 0.003, 0.01, 0.03
- Dataset: COCO val2017, `./val2017` + `./annotations/instances_val2017.json`
- NUM_IMAGES: 100 per variant (quick validation), scale to 500 after verifying
- Eval: pycocotools COCOeval, mAP@[0.5:0.95]
- Inference: prompt=`<OD>`, beams=5, max_tokens=512, repetition_penalty=1.8, length_penalty=1.0
- Scores: heuristic `0.6 + 0.2*area_ratio + 0.15*(1-center_dist)` range [0.60, 0.98]
- Labels: `FLORENCE_TO_COCO` dict (~70 mappings) + `_map_label()` with case-insensitive fallback
- NMS: class-aware, IoU threshold=0.5
- GPU: `NUM_GPUS` in cell 2, BEFORE `import torch`. Uses nvidia-smi subprocess to pick GPUs by free memory.
- Env: conda `vlm_ftune`, PyTorch 2.6.0+cu118, Transformers 4.51.0
- Net Gain = defended_mAP - max(clean+defense_mAP, attacked_mAP). Positive = defense helps.

## Phase 2 Defense Distribution (12 defenses across 3 variants)

### Variant A -- Classical Squeezing (Xu et al. NDSS 2018 family)
Cheapest, fastest. Expected to be weakest based on Guo et al. ICLR 2018 ("weak").
1. JPEG Compression (q=75) -- Dziugaite et al. 2016
2. Gaussian Blur (sigma=1.0) -- Xu et al. NDSS 2018
3. Median Filter (3x3) -- Xu et al. NDSS 2018
4. Bit Depth Reduction (4-bit) -- Xu et al. NDSS 2018

### Variant B -- Advanced Denoising (strongest individual defenses)
Expected best individual performers based on competition results.
1. TVM (w=0.05) -- Guo et al. ICLR 2018, rated "very effective"
2. NLM (h=6) -- Buades et al. 2005; Won CAAD 2018 (Xie et al. CVPR 2019)
3. SVD Spectral Filter (90%) -- per-channel SVD truncation
4. Random Resize + Pad -- Xie et al. ICLR 2018, #2/107 NIPS 2017

### Variant C -- Combined Pipelines (stacked transforms)
Guo et al. ICLR 2018 showed stacked transforms outperform individual.
1. JPEG -> TVM -> NLM (compress + smooth + denoise)
2. Blur -> TVM (spatial + variational smoothing)
3. NLM -> Random Resize (denoise + stochastic)
4. Full Pipeline: JPEG -> TVM -> NLM -> Random Resize (maximum defense)

### DROPPED defenses (no OD paper backing):
- ~~DiffPure~~ -- Nie et al. ICML 2022: classification only (CIFAR-10, ImageNet), never OD
- ~~SmoothVLM~~ -- Sun et al. 2024: VLM text generation/safety only, never OD

## Research References (Attacks)
- **FGSM**: Goodfellow et al. ICLR 2015, "Explaining and Harnessing Adversarial Examples"
- **PGD**: Madry et al. ICLR 2018, "Towards Deep Learning Models Resistant to Adversarial Attacks" -- iterative FGSM with random start, stronger than single-step FGSM
- **C&W**: Carlini & Wagner IEEE S&P 2017 -- optimization-based, bypasses many defenses
- **AutoAttack**: Croce & Hein ICML 2020 -- ensemble of attacks, gold standard for robustness eval
- **Patch attacks**: Brown et al. 2017 -- adversarial patches, physical-world threat

## Research References (Defenses)
- **Guo et al. ICLR 2018** "Countering Adversarial Images using Input Transformations" -- DEFINITIVE ranking: TVM and image quilting "very effective", JPEG and bit-depth "weak"
- **Xie et al. ICLR 2018** "Mitigating Adversarial Effects Through Randomization" -- random resize+pad, #2/107 NIPS 2017 defense competition. Stochastic = harder for attacker.
- **Xie et al. CVPR 2019** "Feature Denoising for Improving Adversarial Robustness" -- NLM-based denoising WON CAAD 2018 defense competition
- **Xu et al. NDSS 2018** "Feature Squeezing" -- JPEG, bit-depth, spatial smoothing. Simple baseline.
- **Nie et al. ICML 2022** DiffPure -- diffusion purification, classification ONLY (not OD)
- **Sun et al. 2024** SmoothVLM -- randomized smoothing for VLMs, text generation ONLY (not OD)
- **Madry et al. ICLR 2018** -- adversarial training, gold standard but requires fine-tuning
- **Realistic OD defense recovery**: 3-12% mAP recovery typical per 2025 autoencoder OD defense papers

## Phase 1 Results (OLD code, for reference)
- Clean mAP = 0.297
- FGSM attacked mAP = 0.226 (eps=0.01)
- FGSM defended mAP = 0.242 (22.5% recovery of 0.071 drop)
- PGD attacked mAP = 0.166
- PGD defended mAP = 0.194 (21.4% recovery of 0.131 drop)
- NOTE: Phase 1 had methodological bugs (preemptive defense, biased scores, prompt ensemble mixing)

## Next Steps
1. **RE-RUN all 3 variants** with fixed eval loop (FGSM attack now actually executes)
2. Compare results across A/B/C to find best defense(s)
3. Scale winning defense(s) to 500 images
4. PGD attack notebook (same defense set, iterative attack)
5. Cross-task transfer experiments (future)

## Future Defense Ideas (if time allows)
- **MirrorCheck** (Fares et al. ICLR 2025) -- cross-modal consistency, detection-specific
- **PuriFlow** (ICCV 2025) -- SR + diffusion, outperforms DiffPure
- **Adversarial Training** (Madry 2018, MMCoA 2024) -- gold standard but needs fine-tuning
- **TPAP** (CVPR 2025) -- test-time pixel purification via FGSM overfitting
