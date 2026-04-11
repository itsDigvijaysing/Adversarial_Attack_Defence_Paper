# LLM Memory -- Florence-2 Adversarial Robustness Project

## Project State
- Phase 1 DONE: FGSM + PGD on Florence-2-Base, COCO val2017. Old defense had critical bugs (see below). Paper submitted to APSCON IEEE.
- Phase 2 IN PROGRESS: `phase2_fgsm.ipynb` created with fixed evaluation + 5 principled defenses. Needs to be run.
- Old files in `OLD/` directory. Do NOT edit.

## Phase 1 Defense Bugs (ALL FIXED in phase2_fgsm.ipynb)
1. JPEG before attack (useless) -> now after attack
2. Noise applied 3-4x cumulatively -> single blur
3. Prompt ensemble mixed OD + free-form text -> single `<OD>`
4. Score recomputation biased results -> raw model scores
5. Unfair baseline -> identical pipeline all conditions
6. Clean mAP with defense never measured -> now measured
7. Edge mask on wrong image -> replaced with standard filters

## Current Defenses (Phase 2)
- JPEG (q=75) -- Dziugaite et al., 2016
- Gaussian Blur (sigma=1.0) -- Xu et al., NDSS 2018
- Median Filter (3x3) -- Xu et al., NDSS 2018
- DiffPure (t=100) -- Nie et al., ICML 2022 (requires `pip install diffusers`)
- SVD Spectral Filter (90%) -- Darabi et al., 2025

## Key Technical Details
- Model: Florence-2-Base via HuggingFace transformers
- Attack loss: `model(..., labels=labels).loss`
- Dataset: COCO val2017, path: `./val2017` (annotations: `./annotations/instances_val2017.json`)
- Eval: pycocotools COCOeval for detection, pycocoevalcap for captioning
- Tasks: `<OD>`, `<CAPTION>`, `<DETAILED_CAPTION>`, `<DENSE_REGION_CAPTION>`
- Env: conda `vlm_ftune`, PyTorch 2.6.0+cu118, Transformers 4.51.0
- GPU Control: `NUM_GPUS` in cell 2 (imports cell), BEFORE `import torch`. Enforced via
  `os.environ["CUDA_VISIBLE_DEVICES"]` using nvidia-smi to pick GPU(s) with most free memory.
  Must be set before torch import or it has no effect. Default: NUM_GPUS=1.
- Generation params: `repetition_penalty=1.8, length_penalty=1.0` on ALL generate() calls
- Scores: Heuristic `score = 0.6 + 0.2*area_ratio + 0.15*(1-center_dist)` [0.60-0.98]
- Labels: `FLORENCE_TO_COCO` dict (~70 mappings) + `_map_label()` in cell 10

## File Map
- phase2_fgsm_variant1.ipynb -- FGSM + 5 defenses (JPEG, blur, median, DiffPure, SVD)
- phase2_fgsm_variant2.ipynb -- FGSM + 5 NEW defenses (bit depth, TVM, NLM, rand resize, SmoothVLM)
- plan.md -- Current plan (concise, issues marked fixed)
- future.md -- Long-term roadmap (7 stages)
- OLD/ -- Phase 1 files (DO NOT EDIT)
- results_phase2/ -- Output directory for new results

## phase2_fgsm.ipynb Deep Analysis (analyzed 2026-04-11)

### Notebook Structure (12 sections)
1. Setup/Imports → 2. Config → 3. Model+Data → 4. NMS+Inference → 5. FGSM Attack
6. 5 Defenses → 7. COCO Eval Helper → 8. Main Loop → 9. Eval All → 10. Results Table
11. Visualization → 12. Net Gain Analysis

### FGSM Attack Flow
- Perturbation in NORMALIZED pixel space (ImageNet mean/std)
- Target labels = model's own beam-search predictions on clean image
- Loss = `model(input_ids, pixel_values, labels=target_ids).loss` → backward → sign(grad)
- x_adv = x + ε·sign(∇L), clamped to [-2.5, 2.5], denormalized to PIL
- Effective pixel change ≈ ε × std(≈0.226) × 255
- Epsilons tested: 0.003, 0.01, 0.03

### Inference Pipeline (UNIVERSAL for all conditions)
- Prompt: always `<OD>`, beams=5, max_tokens=512
- Scores: uniform 1.0 (Florence-2 OD has no confidence output)
- NMS: class-aware, IoU threshold=0.5
- Output: COCO format [x1, y1, w, h]

### Evaluation Conditions (24 total per image)
- clean, clean+5_defenses, 3_epsilons × (no_defense + 5_defenses) = 1+5+3×6=24
- NUM_IMAGES=500 → 12,000 inference calls + 1,500 FGSM attacks

### Net Gain Formula
- Defense worth it only if: defended_mAP > max(clean+defense_mAP, attacked_mAP)
- Accounts for defense cost on clean images

### Critical Design Choices
- Florence-2 OD gives no scores → all scores=1.0 → mAP = recall/precision via IoU matching
- DiffPure uses ddpm-ema-church-256 (not ideal for COCO but available pretrained)
- SVD is per-channel, keeps top 90% singular values
- FGSM perturbation is in normalized space, not raw pixel space

## CRITICAL: Why Phase 2 Variant 2 Fails (analyzed 2026-04-11)

### OLD code (F2_final_fgsm.ipynb): Clean mAP = 0.297, Defended mAP = 0.242
### NEW code (phase2_fgsm_variant2.ipynb): Clean mAP = 0.030, Everything "NOT HELPFUL"

### Root Cause #1: NO SYNTHETIC SCORES (biggest impact)
- **OLD**: Heuristic confidence scores based on box area + center distance:
  `score = 0.6 + 0.2 * area_ratio + 0.15 * (1 - center_dist)` → range [0.60, 0.98]
  COCO mAP REQUIRES varying scores to build precision-recall curves.
- **NEW**: All scores hardcoded to 1.0. COCO eval degenerates — can't rank detections.
- **Impact**: This alone explains most of the 10x mAP drop (0.297 → 0.030).

### Root Cause #2: NO FLORENCE-2 → COCO LABEL MAPPING (55% detections silently dropped)
- Florence-2 outputs labels like: `television`, `mobile phone`, `man`, `houseplant`, 
  `kitchen & dining room table`, `computer keyboard`, `ski`, `studio couch`, `footwear`, etc.
- COCO expects: `tv`, `cell phone`, `person`, `potted plant`, `dining table`, 
  `keyboard`, `skis`, `couch`, (no footwear equivalent), etc.
- `category_mapping.get(label)` returns None → detection silently discarded.
- Tested on 20 images: 233 total labels, only 105 matched (45%), 128 dropped (55%).
- OLD code had same issue BUT mitigated by generation params (see #3).

### Root Cause #3: MISSING GENERATION PARAMETERS
- **OLD**: `model.generate(..., repetition_penalty=1.8, length_penalty=1.0)`
  These constrain Florence-2 to produce more standard, COCO-like category names.
- **NEW**: Neither parameter set (uses defaults).
  Florence-2 outputs broader vocabulary with many non-COCO names.

### Root Cause #4: DiffPure uses WRONG diffusion model
- Uses `google/ddpm-ema-church-256` — trained on church images only.
- Applying to COCO images completely destroys them: clean+diffpure mAP = 0.0001.
- Need an ImageNet-trained or general-purpose diffusion model instead.

### Why defenses all show "NOT HELPFUL"
1. Baseline mAP=0.030 means attack damage is tiny (0.003-0.007 absolute drop)
2. eps=0.01 attacked mAP (0.0258) > eps=0.003 (0.0256) — differences are just noise
3. Defense distortion cost > attack damage at this scale
4. DiffPure destroys images (mAP→0.0001), SmoothVLM too aggressive (mAP→0.007)
5. Net gain = defended - max(clean+defense, attacked) is always negative

### OLD Code's Defense Strategy (what actually worked)
The OLD defenses were fundamentally different from the new ones:
1. **Preemptive defense** (JPEG q=75 + bit quantization BEFORE attack)
2. **Gradient attenuation** (suppress high-variance gradient regions during FGSM)
3. **Gaussian noise injection** (adaptive, frequency-aware, in normalized pixel space)
4. **Spatial smoothing** (edge-preserving hybrid avg+median filter, in normalized space)
5. **Combined defense pipeline** (noise → smoothing → quantization → channel mixing → dropout)
6. **Prompt ensemble** (multi-prompt + multi-augmentation with consensus voting)
All applied as INTEGRATED pipeline, not simple post-hoc PIL transforms.
Result: FGSM mAP recovered from 0.226 → 0.242 (22.5% recovery of 0.071 drop).
PGD mAP recovered from 0.166 → 0.194 (21.4% recovery of 0.131 drop).

### FIX PRIORITY (in order)
1. **Add synthetic scores** (copy OLD formula: area_ratio + center_distance heuristic)
2. **Add Florence→COCO label mapping** dictionary (television→tv, man→person, etc.)
3. **Add `repetition_penalty=1.8, length_penalty=1.0`** to all generate() calls
4. **Replace DiffPure model** with natural-image diffusion model
5. **Reconsider defense approach** — OLD's integrated defense outperformed simple PIL transforms

## Next Steps
1. Fix the 3 critical issues in phase2_fgsm_variant2.ipynb (scores, labels, gen params)
2. Re-run with fixes and verify clean mAP ≈ 0.29
3. Cross-task transfer notebook
4. PGD attack notebook
5. C&W attack notebook

## Additional Defenses Found (web search 2026-04-11)
### Easy to add (PIL→PIL, same API as current defenses):
- Bit Depth Reduction (Xu et al. 2018) -- quantize 8-bit to 4-bit
- Total Variance Minimization (Guo et al. ICLR 2018) -- `skimage.restoration.denoise_tv_chambolle`
- Random Resize + Padding (Xie et al. ICLR 2018) -- stochastic defense
- Non-Local Means Denoising (Buades et al. 2005 / OpenCV) -- `cv2.fastNlMeansDenoisingColored`

### Moderate effort (VLM-specific, recent):
- SmoothVLM (Sun et al. 2024) -- randomized smoothing + majority voting for VLMs
- MirrorCheck (Fares et al. ICLR 2025) -- cross-modal consistency check (detection only)
- DPS (2025) -- partial-perception supervision, training-free
- Denoising Autoencoder -- plug-and-play, needs training on clean COCO

### Advanced (high effort, state-of-art):
- PuriFlow (ICCV 2025) -- SR + diffusion, outperforms DiffPure
- ZeroPur (2025) -- zero-shot purification
- Adversarial Training (Madry 2018, MMCoA 2024) -- fine-tune on adversarial examples
- TPAP (CVPR 2025) -- test-time pixel-level purification via FGSM overfitting
- Attack-as-Defense (ACL 2025) -- protective perturbations

