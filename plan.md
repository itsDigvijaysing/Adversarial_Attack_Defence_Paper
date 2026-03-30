# Current Research Plan: Adversarial Robustness of Florence-2

## Anchor Paper

**CRAFT** -- Zhao et al., "One Object, Multiple Lies: A Benchmark for Cross-task Adversarial Attack on Unified Vision-Language Models," arXiv:2507.07709, 2025. Directly targets Florence-2, cross-task attacks, CrossVLAD benchmark on MSCOCO.

---

## Phase 1 Defense Issues (All Fixed in `phase2_fgsm.ipynb`)

| # | Bug | Status |
|---|-----|--------|
| 1 | JPEG applied before attack (useless) | FIXED -- now applied after attack |
| 2 | Gaussian noise injected 3-4x cumulatively | FIXED -- single Gaussian blur, fixed sigma |
| 3 | Prompt ensemble mixed `<OD>` with free-form text | FIXED -- single `<OD>` prompt only |
| 4 | Score recomputation `0.6+0.2*area+0.15*center` | FIXED -- raw model scores used |
| 5 | Unfair baseline (multi-scale + NMS extras) | FIXED -- identical pipeline for all conditions |
| 6 | Clean mAP with defense never measured | FIXED -- measured for every defense |
| 7 | Edge mask computed on wrong image | FIXED -- replaced with standard median/blur filters |

---

## What's Done

**`phase2_fgsm.ipynb`** -- Complete FGSM evaluation with 5 principled defenses:

| Defense | Reference | Type |
|---------|-----------|------|
| JPEG Compression (q=75) | Dziugaite et al., 2016 | Input preprocessing |
| Gaussian Blur (sigma=1.0) | Xu et al., NDSS 2018 | Input preprocessing |
| Median Filter (3x3) | Xu et al., NDSS 2018 | Input preprocessing |
| DiffPure (t=100) | Nie et al., ICML 2022 | Diffusion purification |
| SVD Spectral Filter (keep 90%) | Darabi et al., arXiv:2502.14976, 2025 | Spectral filtering |

Fair evaluation: all conditions use identical inference (single `<OD>`, standard NMS, raw scores). Defense cost on clean images is measured. Net gain analysis included.

---

## What's Next

### Stage 1: Run `phase2_fgsm.ipynb` and Analyze Results

- Run with `NUM_IMAGES=50` first (debug), then 500 (quick val), then full.
- Install `diffusers` for DiffPure: `pip install diffusers`
- Analyze: which defense has the best net gain? Which costs too much on clean images?

### Stage 2: PGD + C&W Attacks

Create `phase2_pgd.ipynb` and `phase2_cw.ipynb` following the same structure.

- **PGD**: Iterative FGSM (Madry et al., ICLR 2018). Multi-step, stronger than FGSM.
- **C&W**: Optimization-based (Carlini & Wagner, IEEE S&P 2017). Strongest against input-transformation defenses.
- Same 5 defenses, same fair evaluation protocol.

### Stage 3: Cross-Task Transfer

Create `phase2_cross_task.ipynb`:

1. Take FGSM adversarial images (from Stage 1).
2. Run through Florence-2 with captioning prompts (`<CAPTION>`, `<DETAILED_CAPTION>`, `<DENSE_REGION_CAPTION>`).
3. Measure CIDEr/BLEU-4 degradation vs clean captions.
4. Apply best defense(s) from Stage 1 and measure recovery on captioning tasks.
5. Optionally: craft multi-task attack (sum of OD + captioning losses) per CRAFT paper.

Key insight: input-level defenses (JPEG, DiffPure, SVD) should generalize across tasks since they purify the image regardless of downstream task.

### Stage 4: Cross-Model Comparison (If Time Permits)

Pick one small VLM (Qwen2-VL-2B or LLaVA-1.5-7B). Run same FGSM attack + best defense. Compare robustness with Florence-2.

- Reference: Fox et al., arXiv:2512.17902, 2025; La Torre, arXiv:2603.16960, 2026.

---

## Priority Order

1. Run `phase2_fgsm.ipynb` (Stage 1) -- immediate
2. Cross-task transfer (Stage 3) -- high novelty
3. PGD attack (Stage 2) -- straightforward extension
4. C&W attack (Stage 2) -- stronger attack
5. Cross-model (Stage 4) -- if time permits

---

## References

1. Goodfellow et al., "Explaining and Harnessing Adversarial Examples," ICLR 2015.
2. Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," ICLR 2018.
3. Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks," IEEE S&P 2017.
4. Dziugaite et al., "A Study of the Effect of JPG Compression on Adversarial Images," 2016.
5. Xu et al., "Feature Squeezing: Detecting Adversarial Examples in DNNs," NDSS 2018.
6. Nie et al., "Diffusion Models for Adversarial Purification," ICML 2022.
7. Darabi et al., "EigenShield," arXiv:2502.14976, 2025.
8. Zhao et al., "One Object, Multiple Lies," arXiv:2507.07709, 2025.
9. Fu & Zhang, "Adversarial Defense in VLMs: An Overview," arXiv:2601.12443, 2026.
10. Fox et al., "Adversarial Robustness of Vision in Open Foundation Models," arXiv:2512.17902, 2025.
11. Xiao et al., "Florence-2," CVPR 2024.
