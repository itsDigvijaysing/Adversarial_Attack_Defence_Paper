# Research Extension Plan: Adversarial Robustness of Vision Foundation Models

## Critical Assessment of Existing Work

### What Was Done (Phase 1 - Completed)

The initial phase evaluated Florence-2-Base's adversarial robustness for object detection on COCO val2017 (5,000 images) using FGSM and PGD attacks. A multi-layered inference-time defense (JPEG compression + Gaussian noise + spatial smoothing + prompt ensemble + NMS) was proposed, recovering 21.6%--40.1% of lost mAP. A paper was submitted to APSCON IEEE.

### Strengths of Existing Work

- **First quantitative study** of Florence-2's adversarial robustness for object detection.
- **Practical defense**: no model retraining required, purely inference-time.
- **Solid evaluation protocol**: standard COCO metrics (mAP, AP50, AP75) via pycocotools.
- **Reproducible**: full notebooks with end-to-end pipelines.

### Weaknesses and Gaps (Honest Critique)

| Gap | Why It Matters |
|-----|----------------|
| **Only FGSM + PGD attacks** | These are well-known, relatively simple gradient attacks. Reviewers expect stronger adversaries (C&W, AutoAttack). See Croce & Hein, ICML 2020. |
| **No adaptive attack evaluation** | The defense was never tested against an attacker who *knows* the defense. This is a fundamental requirement per Tramer et al., 2020 ("On Adaptive Attacks to Machine Learning Defenses"). |
| **No defense baselines** | The multi-layered defense is not compared against any other defense method (adversarial training, diffusion purification, spectral filtering). Without baselines, the 21--40% recovery is hard to contextualize. |
| **Single model only** | No cross-model comparison. We cannot claim whether Florence-2 is more or less robust than alternatives like LLaVA, Qwen2-VL, or BLIP-2. |
| **No cross-task evaluation** | Florence-2 is a unified multi-task model, but only object detection was tested. Adversarial examples may transfer across tasks (captioning, grounding, segmentation). Zhao et al. (arXiv:2507.07709) demonstrated this exact cross-task vulnerability. |
| **Defense recovery is modest** | 21--40% recovery leaves 60--79% of performance still lost. This is insufficient for safety-critical applications. |
| **No transferability analysis** | Do adversarial examples crafted for Florence-2 fool other VLMs? This is a key question for real-world threat modeling. |
| **Prompt ensemble is ad-hoc** | The 4-prompt ensemble lacks theoretical justification. No analysis of how many prompts are needed or which prompts are optimal. |

---

## What New Research Has Emerged (2025--2026)

### Key Papers Directly Relevant to This Work

1. **Cross-Task Adversarial Attacks on Unified VLMs (CRAFT)**
   - Zhao et al., "One Object, Multiple Lies: A Benchmark for Cross-task Adversarial Attack on Unified Vision-Language Models," arXiv:2507.07709, 2025.
   - Introduces CRAFT attack and CrossVLAD benchmark. Directly targets Florence-2 and demonstrates that a single perturbation can fool multiple tasks simultaneously.
   - **Relevance**: Directly extends our work by showing Florence-2's cross-task vulnerability.

2. **Adversarial Defense in VLMs: An Overview**
   - Fu & Zhang, "Adversarial Defense in Vision-Language Models: An Overview," arXiv:2601.12443, 2026.
   - Categorizes defenses into training-time, test-time adaptation, and training-free. Our defense falls under training-free.
   - **Relevance**: Provides the taxonomy to position our defense and identify alternatives to compare against.

3. **EigenShield: Spectral Filtering Defense**
   - Darabi et al., "EigenShield: Causal Subspace Filtering via Random Matrix Theory for Adversarially Robust Vision-Language Models," arXiv:2502.14976, 2025.
   - Uses Random Matrix Theory to detect and filter adversarial noise from embeddings at inference time. Attack-agnostic and architecture-independent.
   - **Relevance**: A principled alternative to our heuristic-based defense. Strong comparison baseline.

4. **X-Transfer: Super Transferable Attacks on CLIP**
   - Huang et al., "X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP," arXiv:2505.05528, ICML 2025.
   - Demonstrates Universal Adversarial Perturbations (UAPs) that transfer across data, domains, models, and tasks.
   - **Relevance**: Florence-2 uses a CLIP-like vision encoder. These transferable attacks are a realistic threat.

5. **Double Visual Defense**
   - Wang et al., "Double Visual Defense: Adversarial Pre-training and Instruction Tuning for Improving Vision-Language Model Robustness," arXiv:2501.09446, 2025.
   - Achieves ~20% robustness improvement on ImageNet via adversarial pre-training of CLIP + adversarial instruction tuning of LLaVA.
   - **Relevance**: Represents the training-time defense approach. Useful as an upper-bound reference for what training-based methods can achieve.

6. **Adversarial Robustness of Vision in Open Foundation Models**
   - Fox et al., "Adversarial Robustness of Vision in Open Foundation Models," arXiv:2512.17902, IEEE Access, 2025.
   - Evaluates LLaVA-1.5-13B and Llama 3.2 Vision-8B under PGD attacks. Finds robustness is independent of standard benchmark accuracy.
   - **Relevance**: Provides cross-model comparison methodology we should follow.

7. **Chain of Attack (CVPR 2025)**
   - Xie et al., "Chain of Attack: On the Robustness of Vision-Language Models Against Transfer-Based Adversarial Attacks," CVPR 2025.
   - Proposes iterative multi-modal semantic-aware adversarial attack with Targeted Contrastive Matching.
   - **Relevance**: State-of-the-art transfer attack method relevant to our transferability analysis.

8. **FARE: Robust CLIP Fine-Tuning (ICML 2024)**
   - Schlarmann et al., "Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models," ICML 2024. Code: https://github.com/chs20/RobustVLM
   - Unsupervised adversarial fine-tuning of CLIP's vision encoder. Achieves robust embeddings without task-specific supervision.
   - **Relevance**: If Florence-2's vision encoder can be swapped or fine-tuned with FARE, this provides a principled training-time defense.

9. **Diffusion-Based Adversarial Purification**
   - Nie et al., "Diffusion Models for Adversarial Purification," ICML 2022 (DiffPure). Wu et al., "ADBM: Adversarial Diffusion Bridge Model," ICLR 2025.
   - Uses diffusion forward + reverse process to purify adversarial images. ADBM improves over DiffPure by 4.4% on CIFAR-10.
   - **Relevance**: Diffusion purification is a strong training-free defense baseline to compare against our approach.

10. **Adversarial Attacks on VLMs (Survey, 2026)**
    - La Torre, "Adversarial attacks against Modern Vision-Language Models," arXiv:2603.16960, 2026.
    - Evaluates LLaVA-v1.5-7B and Qwen2.5-VL-7B under BIM, PGD, and CLIP-based spectral attacks. Qwen2.5-VL shows significantly stronger robustness (6.5--15.5% ASR vs 52.6--66.9% for LLaVA).
    - **Relevance**: Provides attack success rates for direct comparison with our Florence-2 results.

11. **Defenses Against Adversarial Attacks on Object Detection (Survey)**
    - MDPI Information, "Defenses Against Adversarial Attacks on Object Detection: Methods and Future Directions," 2025.
    - Categorizes defenses into preprocessing, adversarial training, detection, architectural, ensemble, and certified defenses.
    - **Relevance**: Comprehensive taxonomy for positioning our defense contributions.

12. **AutoAttack Benchmark**
    - Croce & Hein, "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks," ICML 2020.
    - Standard benchmark for adversarial robustness evaluation. Combines APGD-CE, APGD-T, FAB, and Square attacks.
    - **Relevance**: Industry-standard evaluation tool that should be used to validate any robustness claim.

---

## Proposed Extension Plan

### Stage 1: Stronger and Diverse Attack Evaluation

**Objective**: Evaluate Florence-2 under a broader, more rigorous set of attacks beyond FGSM/PGD.

**Tasks**:
- Implement C&W (Carlini-Wagner) L2 attack using `torchattacks` library. C&W is an optimization-based attack that is significantly stronger than FGSM/PGD for bypassing heuristic defenses.
  - Reference: Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks," IEEE S&P, 2017.
- Run AutoAttack (APGD-CE + APGD-T + FAB + Square) using the `autoattack` Python package. This is the standard robustness benchmark.
  - Reference: Croce & Hein, ICML 2020.
  - Code: `pip install autoattack`, GitHub: https://github.com/fra31/auto-attack
- Compare attack success rates and mAP degradation across all attacks (FGSM, PGD, C&W, AutoAttack) at matched epsilon budgets.
- Document which attacks Florence-2 is most/least vulnerable to.

**Key Consideration**: AutoAttack is designed for classifiers. For object detection, we may need to adapt the loss function. Use the detection loss (same as used in FGSM/PGD notebooks) as the optimization objective.

**Deliverable**: Table comparing mAP under Clean / FGSM / PGD / C&W / AutoAttack at epsilon = {0.003, 0.01, 0.03}.

---

### Stage 2: Cross-Task Adversarial Transfer Within Florence-2

**Objective**: Test whether adversarial examples crafted for object detection transfer to Florence-2's other tasks.

**Tasks**:
- Use the adversarial images already generated (FGSM/PGD for OD) and run them through Florence-2's other task prompts:
  - `<CAPTION>` (image captioning)
  - `<DETAILED_CAPTION>` (detailed captioning)
  - `<OD>` to `<DENSE_REGION_CAPTION>` (dense region captioning)
  - `<OCR>` (if text is present)
- Measure degradation on each task qualitatively and quantitatively (BLEU/CIDEr for captioning, mAP for detection variants).
- Optionally implement the CRAFT attack from Zhao et al. (arXiv:2507.07709) which specifically optimizes for cross-task fooling.
  - Reference: Zhao et al., arXiv:2507.07709, 2025.

**Deliverable**: Cross-task transfer matrix showing attack success rates when adversarial examples crafted for Task A are evaluated on Tasks B, C, D.

---

### Stage 3: Defense Baselines and Comparison

**Objective**: Compare the existing multi-layered defense against at least two principled defense baselines.

**Tasks**:

**Baseline 1: Diffusion-Based Purification**
- Implement DiffPure-style defense: add noise via diffusion forward process (small number of steps), then denoise.
- Use a lightweight pretrained diffusion model (e.g., from `diffusers` library, a small DDPM on ImageNet).
- Reference: Nie et al., "Diffusion Models for Adversarial Purification," ICML 2022.
- Code reference: https://github.com/NVlabs/DiffPure

**Baseline 2: Spectral Filtering (EigenShield-inspired)**
- Implement a simplified version: compute SVD of image features, filter components that deviate from clean statistics using Marchenko-Pastur law threshold.
- Reference: Darabi et al., arXiv:2502.14976, 2025.

**Baseline 3: Simple Adversarial Training (if compute allows)**
- Fine-tune Florence-2's vision encoder for a few epochs with adversarial examples in the training loop.
- Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," ICLR 2018.

**Comparison Protocol**:
- All defenses tested under identical conditions (same attacks, same epsilon, same dataset subset).
- Metrics: Clean mAP (to measure defense overhead), Defended mAP, Recovery %, Inference time overhead.

**Deliverable**: Comparison table: Our Defense vs. DiffPure vs. Spectral Filtering vs. (Adversarial Training if feasible).

---

### Stage 4: Cross-Model Robustness Comparison

**Objective**: Position Florence-2's robustness relative to other VLMs under identical attack conditions.

**Tasks**:
- Select 1--2 additional models for comparison:
  - **Option A**: Qwen2-VL-2B (lightweight, reportedly robust per La Torre, arXiv:2603.16960)
  - **Option B**: LLaVA-1.5-7B (widely used, well-studied adversarial properties)
- Run the same attacks (at minimum FGSM and PGD) on these models for object detection or VQA.
- Compare robustness metrics side by side.
- Reference: Fox et al., arXiv:2512.17902, 2025; La Torre, arXiv:2603.16960, 2026.

**Key Consideration**: Different VLMs have different architectures and input pipelines. Ensure attacks are adapted to each model's preprocessing (normalization, tokenization). Keep the perturbation budget (epsilon) identical in the pixel domain for fair comparison.

**Deliverable**: Cross-model robustness comparison table + analysis of architectural factors influencing robustness.

---

### Stage 5: Adaptive Attack Evaluation

**Objective**: Evaluate the defense under an adaptive attacker who is fully aware of the defense pipeline.

**Tasks**:
- Implement an adaptive PGD attack that incorporates the defense transformations into the attack optimization loop:
  - Include JPEG compression (use differentiable JPEG approximation, e.g., `DiffJPEG` library).
  - Include Gaussian noise (use expectation over transformation, EoT).
  - Include spatial smoothing in the forward pass.
- Reference: Tramer et al., "On Adaptive Attacks to Machine Learning Defenses," arXiv:2002.09532, 2020. This is the gold-standard paper on evaluating defenses.
- Reference for EoT: Athalye et al., "Synthesizing Robust Adversarial Examples," ICML 2018.
- Measure how much the defense degrades under adaptive attacks vs. non-adaptive attacks.

**Why This Is Critical**: A defense that only works against non-adaptive attacks provides a false sense of security. Reviewers of top venues will immediately ask this question.

**Deliverable**: Table showing defense performance under non-adaptive vs. adaptive attacks.

---

### Stage 6: Transferability Analysis

**Objective**: Study whether adversarial examples transfer across different VLMs.

**Tasks**:
- Generate adversarial examples on Florence-2 (source model).
- Evaluate them on 1--2 other VLMs (target models) without further optimization.
- Reverse direction: generate on other VLMs, evaluate on Florence-2.
- Measure transfer attack success rates.
- Reference: Xie et al., CVPR 2025 (Chain of Attack); Huang et al., ICML 2025 (X-Transfer).

**Deliverable**: Transfer matrix showing attack success rates between model pairs.

---

### Stage 7: Consolidation, Ablation, and Paper Update

**Objective**: Compile all results, run ablations, and produce a publication-ready update.

**Tasks**:
- **Ablation study** of the multi-layered defense: measure contribution of each component (JPEG alone, noise alone, smoothing alone, ensemble alone, all combined).
- **Statistical significance**: run key experiments 3 times with different random seeds, report mean and standard deviation.
- Update the research paper with:
  - Expanded related work section citing papers listed above.
  - New experimental results (stronger attacks, defense baselines, cross-model, adaptive).
  - Revised claims based on adaptive attack results.
  - Proper comparison tables.
- Update README with final results and instructions.

**Deliverable**: Updated paper draft, updated repository, final comparison tables.

---

## Priority Order (If Time Is Limited)

If not all stages can be completed, prioritize in this order:

1. **Stage 1** (Stronger attacks) -- Most critical gap. A few days with existing infrastructure.
2. **Stage 5** (Adaptive attacks) -- Second most critical. Reviewers will ask for this.
3. **Stage 3** (Defense baselines) -- Needed to contextualize the defense.
4. **Stage 2** (Cross-task transfer) -- Unique to Florence-2, high novelty.
5. **Stage 7** (Ablation + paper) -- Needed for publication.
6. **Stage 4** (Cross-model comparison) -- Nice to have, strengthens the paper.
7. **Stage 6** (Transferability) -- Interesting but lowest priority.

---

## Tools and Libraries Required

| Tool | Purpose | Install |
|------|---------|---------|
| `torchattacks` | C&W, PGD, FGSM, BIM implementations | `pip install torchattacks` |
| `autoattack` | AutoAttack benchmark | `pip install git+https://github.com/fra31/auto-attack` |
| `diffusers` | Pretrained diffusion models for purification | `pip install diffusers` |
| `DiffJPEG` | Differentiable JPEG for adaptive attacks | https://github.com/mlomnitz/DiffJPEG |
| `transformers` | Florence-2, LLaVA, Qwen2-VL model loading | Already installed |
| `pycocotools` | COCO evaluation metrics | Already installed |

---

## References

1. Carlini, N. & Wagner, D. "Towards Evaluating the Robustness of Neural Networks." IEEE S&P, 2017.
2. Croce, F. & Hein, M. "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks." ICML, 2020.
3. Tramer, F. et al. "On Adaptive Attacks to Machine Learning Defenses." arXiv:2002.09532, 2020.
4. Athalye, A. et al. "Synthesizing Robust Adversarial Examples." ICML, 2018.
5. Madry, A. et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR, 2018.
6. Nie, W. et al. "Diffusion Models for Adversarial Purification." ICML, 2022.
7. Zhao, J. et al. "One Object, Multiple Lies: Cross-task Adversarial Attack on Unified VLMs." arXiv:2507.07709, 2025.
8. Fu, X. & Zhang, L. "Adversarial Defense in Vision-Language Models: An Overview." arXiv:2601.12443, 2026.
9. Darabi, N. et al. "EigenShield: Causal Subspace Filtering via Random Matrix Theory." arXiv:2502.14976, 2025.
10. Huang, H. et al. "X-Transfer Attacks: Super Transferable Adversarial Attacks on CLIP." ICML, 2025. arXiv:2505.05528.
11. Wang, Z. et al. "Double Visual Defense." arXiv:2501.09446, 2025.
12. Fox, J. et al. "Adversarial Robustness of Vision in Open Foundation Models." IEEE Access, 2025. arXiv:2512.17902.
13. Xie et al. "Chain of Attack: On the Robustness of VLMs Against Transfer-Based Adversarial Attacks." CVPR, 2025.
14. La Torre, A.P. "Adversarial attacks against Modern Vision-Language Models." arXiv:2603.16960, 2026.
15. Schlarmann, C. et al. "Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings." ICML, 2024.
16. Wu et al. "ADBM: Adversarial Diffusion Bridge Model." ICLR, 2025.
17. MDPI. "Defenses Against Adversarial Attacks on Object Detection: Methods and Future Directions." Information, 2025.
18. Xiao, B. et al. "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks." CVPR, 2024.
