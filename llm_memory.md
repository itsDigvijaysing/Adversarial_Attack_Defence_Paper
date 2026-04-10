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
- Dataset: COCO val2017, path: `./Dataset/coco/`
- Eval: pycocotools COCOeval for detection, pycocoevalcap for captioning
- Tasks: `<OD>`, `<CAPTION>`, `<DETAILED_CAPTION>`, `<DENSE_REGION_CAPTION>`
- Env: conda `vlm_ftune`, PyTorch 2.6.0+cu118, Transformers 4.51.0

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

## Next Steps
1. Run phase2_fgsm.ipynb (NUM_IMAGES=50 debug, then 500, then full)
2. Cross-task transfer notebook
3. PGD attack notebook
4. C&W attack notebook

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

