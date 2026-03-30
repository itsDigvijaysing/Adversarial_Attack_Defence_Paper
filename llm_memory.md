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
- phase2_fgsm.ipynb -- NEW: FGSM + 5 defenses, fair evaluation
- plan.md -- Current plan (concise, issues marked fixed)
- future.md -- Long-term roadmap (7 stages)
- OLD/ -- Phase 1 files (DO NOT EDIT)
- results_phase2/ -- Output directory for new results

## Next Steps
1. Run phase2_fgsm.ipynb (NUM_IMAGES=50 debug, then 500, then full)
2. Cross-task transfer notebook
3. PGD attack notebook
4. C&W attack notebook
