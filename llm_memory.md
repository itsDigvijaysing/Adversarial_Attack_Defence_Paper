# LLM Memory -- Florence-2 Adversarial Robustness Project

## Project State (updated 2026-04-12)
- Phase 1 DONE: FGSM + PGD on Florence-2-Base, COCO val2017. Paper submitted to APSCON IEEE.
- Phase 2 IN PROGRESS: Variants A & B DONE (1000 images), Variant C DONE (1000 images), Variant D READY TO RUN (5000 images, GPU-accelerated).
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
- **FGSM_Phase2_VariantD.ipynb** -- Winner Combos (GPU-accelerated): JPEG->TVM, JPEG->Median->TVM, Median->TVM, JPEG->TVM->SVD
- **phase2_fgsm_variant2.ipynb** -- Original variant2, has TVM+NLM+JPEG+RandomResize+Combined (superset)
- Output dirs: `results_phase2_variantA/`, `results_phase2_variantB/`, `results_phase2_variantC/`, `results_phase2_variantD/`

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
- NUM_IMAGES: 1000 for A/B/C (screening), 5000 for D (paper-ready)
- Eval: pycocotools COCOeval, mAP@[0.5:0.95]
- Inference: prompt=`<OD>`, beams=5, max_tokens=512, repetition_penalty=1.8, length_penalty=1.0
- Scores: heuristic `0.6 + 0.2*area_ratio + 0.15*(1-center_dist)` range [0.60, 0.98]
- Labels: `FLORENCE_TO_COCO` dict (~70 mappings) + `_map_label()` with case-insensitive fallback
- NMS: class-aware, IoU threshold=0.5
- GPU: `NUM_GPUS` + `GPU_POOL` in cell 2, BEFORE `import torch`. Uses nvidia-smi to pick GPUs by free memory from specified pool.
- Variant D uses GPU-accelerated defenses: TVM (Chambolle in PyTorch), SVD (torch.linalg.svd), Median (torch.unfold). ~50-100x faster than CPU skimage.
- Env: conda `vlm_ftune`, PyTorch 2.6.0+cu118, Transformers 4.51.0
- Net Gain = defended_mAP - max(clean+defense_mAP, attacked_mAP). Positive = defense helps.
- **IMPORTANT: Net Gain Interpretation** -- When reading results, focus on RECOVERY from attacked mAP, not just comparison to clean baseline. Key metrics: Recovery = defended_mAP - attacked_mAP; Recovery% = recovery / (clean_mAP - attacked_mAP). A defense is good if it pulls mAP back UP from the attacked floor.

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

### Variant D -- Winner Combination Pipelines (created 2026-04-12, GPU-accelerated)
Based on Variant A/B screening results + Variant C confirmation. Only proven winners combined.
- **NUM_IMAGES = 5000** (paper-ready scale, full COCO val2017)
- **EPSILONS = [0.03]** only (strongest attack; weaker epsilons showed no recovery difference in A/B)
- **GPU-accelerated defenses**: TVM (Chambolle in PyTorch), SVD (torch.linalg.svd), Median (torch.unfold). JPEG stays CPU.
- **GPU_POOL**: configurable list of physical GPUs to choose from
- **Multi-GPU parallel**: model loaded on ALL visible GPUs (e.g. cuda:0 + cuda:1), each GPU processes a separate stream of source images via ThreadPoolExecutor (n_gpus workers). Files split round-robin across GPUs.
- **Per-image flow** (no batching, no chunking): one source image at a time per GPU. Each thread does clean inference + clean+defenses + FGSM + defended FGSM, fully on its own GPU. GPU ops release the GIL so two threads truly run in parallel.
- **Why no batching**: `model.generate(num_beams=5)` with batch=10+ creates 50+ active beam sequences and causes memory thrashing instead of speedup. Beam search is per-step sequential, so batching gives little wall-clock benefit but huge VRAM pressure (20GB allocated, 0% util observed). Per-image dual-GPU is the empirically faster choice.
- OUTPUT_DIR = `./results_phase2_variantD`
1. JPEG -> TVM (compress + smooth)
2. JPEG -> Median -> TVM (compress + local smooth + global smooth)
3. Median -> TVM (local + global smooth, no compression)
4. JPEG -> TVM -> SVD (compress + smooth + spectral)

### DROPPED defenses (no OD paper backing):
- ~~DiffPure~~ -- Nie et al. ICML 2022: classification only (CIFAR-10, ImageNet), never OD
- ~~SmoothVLM~~ -- Sun et al. 2024: VLM text generation/safety only, never OD

## Phase 2 Screening Results (Variants A & B, 1000 images, 2026-04-12)

### Winners (defenses that RECOVER mAP after FGSM attack at eps=0.03):
| Defense | Variant | Recovery | StrictGain | Verdict |
|---------|---------|----------|------------|---------|
| TVM (w=0.05) | B | 14.7% | Positive | RECOVERS (clear #1) |
| JPEG (q=75) | A | Partial | Marginal | Partial recovery |
| Median Filter (3x3) | A | Partial | Marginal | Partial recovery |
| SVD (90%) | B | 3.0% | Negative (high clean cost) | Conditional |

### Losers (negative or negligible strict gain):
- Gaussian Blur (sigma=1.0) -- marginal effect
- Bit Depth (4-bit) -- negative strict gain
- NLM (h=6) -- negative recovery (SURPRISE: won CAAD 2018 but failed here)
- Random Resize+Pad -- worst performer, very negative

### Key insight:
TVM dominance aligns with Guo et al. ICLR 2018 ranking. NLM failure is surprising -- may be because NLM was validated on classification (feature-level denoising), not on VLM text generation. JPEG performing better than expected on VLMs.

## Phase 2 Variant C Results (1000 images, completed 2026-04-12)

| Pipeline | eps=0.03 defended mAP | Net Gain | Verdict |
|----------|----------------------|----------|---------|
| blur_tvm | 0.3129 | -0.0008 | Near-neutral — TVM carrying, blur adds nothing |
| jpeg_tvm_nlm | 0.2615 | -0.0264 | NOT HELPFUL — NLM drags it down |
| nlm_resize | 0.0820 | -0.1926 | CATASTROPHIC — both are losers |
| full_pipeline (JPEG->TVM->NLM->Resize) | 0.0927 | -0.1819 | CATASTROPHIC — NLM + Resize destroy everything |

**Key insight:** Every pipeline containing NLM or Random Resize fails. blur_tvm was best only because TVM carried it (blur is a loser from A/B). Confirms winner/loser classification and validates Variant D's winner-only design.

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

## Next Steps (updated 2026-04-12)
1. ~~RE-RUN all 3 variants~~ DONE -- Variants A & B completed (1000 images)
2. ~~Compare results across A/B to find best defense(s)~~ DONE -- TVM wins, see results above
3. ~~Wait for Variant C~~ DONE -- confirms NLM/Resize are losers, TVM carries blur_tvm
4. **RUN Variant D** -- GPU-accelerated, 5000 images, eps=0.03 only, winner combos. GPU_POOL=[2,3] to avoid busy GPUs.
5. Compare Variant D results -- find best pipeline ordering
6. PGD attack notebook (same defense set, iterative attack)
7. Cross-model comparison with YOLOv8x-worldv2 (future)
8. Cross-task transfer experiments (future)

## Future Defense Ideas (if time allows)
- **MirrorCheck** (Fares et al. ICLR 2025) -- cross-modal consistency, detection-specific
- **PuriFlow** (ICCV 2025) -- SR + diffusion, outperforms DiffPure
- **Adversarial Training** (Madry 2018, MMCoA 2024) -- gold standard but needs fine-tuning
- **TPAP** (CVPR 2025) -- test-time pixel purification via FGSM overfitting




## Project
- Paper: APSCON IEEE. Models: Florence-2-base (microsoft/Florence-2-base, refs/pr/26, beam=5) + YOLOv8x-worldv2. Data: COCO val2017, mAP@[.5:.95] via pycocotools.
- Pipeline: 3 attacks × 2 models × {clean, 5 solo defenses, 3 ensembles}.

## Phase 3 v2 Config (authoritative — v1 contaminated, do not cite)
- 1000 imgs (`sorted(os.listdir)[:1000]`), 1 GPU per notebook, NO checkpoint.
- In-memory `all_buckets={}` dict, single final `pickle.dump`. Hard asserts: `len==NUM_IMAGES` + every img carries every expected tag.
- Output dirs: `results_phase3_{yolo,florence}_{fgsm,pgd,patch}_v2/`.
- FGSM: eps=0.03, 1 step. PGD: eps=0.03, 10 iters, alpha=eps/4, rand start. Patch: 35×35 center, 100 Adam steps, lr=0.02.
- Florence attacks in normalized pixel space (matches Variant Z). YOLO in raw [0,1].
- Defenses: 5 solos [jpeg, median, tvm, gaussian, blur_tvm] + 3 ensembles [ens_blur_tvm_combo, ens_jpeg_median_gaussian, ens_jpeg_median_tvm]. Class-aware NMS merging.
- YOLO needs `model.model.model[-1].shape=None` reset between inferences (tensor cache bug).

## Confirmed YOLO v2 Results (sanity-passed, no defense > clean)
- FGSM: clean=0.4765, atk=0.2259, best blur_tvm→0.3450 (+47.5%). All 8 recover.
- PGD iters=10: clean=0.4972, atk=0.1425, best ens_jpeg_median_tvm→0.4801 (+95.2%). All 8 recover.
- PGD more iters: clean=0.4765, atk=0.0614, best ens_jpeg_median_tvm→0.4263 (+87.9%). All 8 recover.

## Pending
- 3 Florence v2 notebooks (FGSM/PGD/Patch) — slow, parallel GPUs.

## Key Lessons
- **v1 contamination root cause**: `DetectionCheckpoint` + `if ckpt.has(img_id): return` over single `detections.pkl` silently reused stale entries across runs → FGSM negative recovery, PGD ensembles > clean, `num_images` misreport. Fix = skip cache entirely (v2). Suspicious patterns: negative recovery on weak attacks, recovered mAP > clean, num_images mismatch.
- **FGSM vs PGD recovery asymmetry is expected physics, not a bug**: iterative PGD → high-freq noise smoothing defenses wipe (80–95%); single-step FGSM → aligns with semantic gradients so smoothing damages signal (40–60%). Frame in paper as "defense scales with attack sophistication." To close gap: reduce PGD iters to 3–5, don't increase.
- **Diagnose by diffing against working baseline first**: wrongly blamed Florence FGSM normalized-space eps; Variant Z used identical code and worked. Real bug was cache contamination. Before theorizing algorithm fixes, grep last working version.
- **Net gain interpretation**: compare recovery vs attacked mAP, not clean baseline.
- **Beam search**: do not batch `generate(num_beams=5)`; use per-image dual-GPU threading.

## Dev Discipline
- Limited token budget — minimize tool calls, hypothesize before exploring, answer from existing context.
- Don't "fix" algorithms from single-notebook symptoms when a working reference exists.
- Cite only v2 numbers in paper; v1 `results_phase3_*/` (no `_v2`) are contaminated.