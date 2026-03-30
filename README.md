# Evaluating and Enhancing the Adversarial Robustness of Florence-2 Model

This repository contains the code, notebooks, research paper, and setup files for evaluating the adversarial robustness of the Florence-2-Base vision foundation model (specifically for object detection) and developing defense strategies against adversarial attacks.

**Author:** Rajput Digvijaysing Bhatesing
**Instructor:** Prof. C. Krishna Mohan
**TA:** Popat Raj Rameshkumar
**Institution:** Indian Institute of Technology Hyderabad

---

## Table of Contents

1.  [Overview](#overview)
2.  [What Has Been Done (Phase 1)](#what-has-been-done-phase-1)
3.  [Key Results (Phase 1)](#key-results-phase-1)
4.  [Known Limitations](#known-limitations)
5.  [Planned Extensions (Phase 2)](#planned-extensions-phase-2)
6.  [Repository Contents](#repository-contents)
7.  [Setup Instructions](#setup-instructions)
8.  [Usage / Running Experiments](#usage--running-experiments)
9.  [References](#references)

---

## Overview

Vision Foundation Models (VFMs) like Florence-2 (Xiao et al., CVPR 2024) demonstrate remarkable capabilities across diverse visual tasks. However, their robustness against adversarial attacks remains under-explored and is critical for real-world deployment. This project provides an empirical study on the adversarial resilience of Florence-2-Base for COCO object detection.

**Phase 1 (Completed):** Evaluated Florence-2-Base under FGSM and PGD attacks. Proposed a multi-layered, inference-time defense pipeline that recovers 21.6%--40.1% of lost performance without model retraining.

**Phase 2 (In Progress):** After identifying critical issues in Phase 1's defense pipeline (see [plan.md](plan.md)), we rebuilt the evaluation with fair baselines and 5 principled defenses backed by published research. The new notebook `phase2_fgsm.ipynb` fixes all known issues and adds DiffPure (Nie et al., ICML 2022) and SVD spectral filtering (Darabi et al., 2025). Next: cross-task transfer, PGD, C&W attacks.

---

## What Has Been Done (Phase 1)

### Attacks Implemented
- **FGSM** (Fast Gradient Sign Method) -- single-step gradient attack at epsilon = {0.001, 0.003, 0.03}
- **PGD** (Projected Gradient Descent) -- iterative multi-step attack at epsilon = {0.007, 0.01, 0.03}, alpha = {0.002, 0.0025}, iterations = {5, 10}

### Defense Pipeline (Inference-Time, No Retraining)
```
Input Image
    |
JPEG Compression (quality=75)
    |
Gaussian Noise Injection (adaptive, sigma=0.01-0.08)
    |
Spatial Smoothing (5x5 Gaussian kernel)
    |
Model Inference with Prompt Ensemble (4 prompt variants)
    |
Non-Maximum Suppression (IoU=0.45)
    |
Final Detections
```

### Dataset
- **COCO 2017 Validation Set** -- 5,000 images with standard bounding box annotations
- Evaluation via pycocotools COCOeval (mAP, AP50, AP75)

---

## Key Results (Phase 1)

- Florence-2-Base shows **significant vulnerability** to both FGSM and PGD attacks, with substantial mAP drops under all tested epsilon values.
- The multi-layered defense recovers **21.6%--40.1%** of the performance lost due to attacks.
- Defense effectiveness scales with attack intensity (higher recovery at stronger attacks).
- The defense adds manageable computational overhead and requires **no model retraining**.

---

## Known Limitations

These are honest assessments of what the current work does not address:

1. **Limited attack diversity**: Only FGSM and PGD were tested. Stronger attacks like C&W (Carlini & Wagner, 2017) and AutoAttack (Croce & Hein, ICML 2020) were not evaluated.
2. **No adaptive attack evaluation**: The defense was not tested against an attacker who knows the defense pipeline (Tramer et al., 2020).
3. **No defense baselines**: The proposed defense is not compared against other methods (diffusion purification, spectral filtering, adversarial training).
4. **Single model**: No cross-model comparison with other VLMs (LLaVA, Qwen2-VL, BLIP-2).
5. **Single task**: Only object detection was evaluated. Florence-2 supports multiple tasks, and cross-task adversarial transfer was not studied.
6. **Modest recovery**: 21--40% recovery leaves the majority of performance still lost.
7. **No ablation study**: Individual contributions of each defense component are not isolated.

---

## Planned Extensions (Phase 2)

The current focused plan is in [plan.md](plan.md). A broader long-term roadmap is in [future.md](future.md). The key stages are:

| Stage | Description | Key Reference |
|-------|-------------|---------------|
| 1 | Stronger attacks (C&W, AutoAttack) | Croce & Hein, ICML 2020 |
| 2 | Cross-task adversarial transfer within Florence-2 | Zhao et al., arXiv:2507.07709, 2025 |
| 3 | Defense baselines (diffusion purification, spectral filtering) | Nie et al., ICML 2022; Darabi et al., arXiv:2502.14976 |
| 4 | Cross-model robustness comparison | Fox et al., arXiv:2512.17902, 2025 |
| 5 | Adaptive attack evaluation | Tramer et al., arXiv:2002.09532, 2020 |
| 6 | Transferability analysis across VLMs | Xie et al., CVPR 2025; Huang et al., ICML 2025 |
| 7 | Ablation, consolidation, paper update | -- |

---

## Repository Contents

```
.
├── phase2_fgsm.ipynb            # [NEW] FGSM attack + 5 principled defenses, fair evaluation
├── plan.md                      # Current research plan (concise, issues marked fixed)
├── future.md                    # Long-term roadmap (cross-model, adaptive attacks, etc.)
├── llm_memory.md                # Project state for LLM context
├── environment.yml              # Full Conda environment specification
├── mini_env.yml                 # Minimal environment specification
├── Presentation.pdf             # Project presentation
├── README.md                    # This file
├── results_phase2/              # Output directory for Phase 2 results
└── OLD/                         # Phase 1 files (preserved, not modified)
    ├── F2_final_fgsm.ipynb      # Phase 1 FGSM notebook (with results)
    ├── F2_final_pgd.ipynb       # Phase 1 PGD notebook (with results)
    ├── F2_Updated.ipynb         # Phase 1 integrated notebook
    ├── F2_Optimized.ipynb       # Phase 1 optimized framework
    ├── F2_Optimized.py          # Phase 1 Python module
    ├── Report.pdf               # Research report
    ├── MCA Presentation - 1.pdf
    ├── MCA Presentation - 2.pdf
    └── Adversarial_Attack_Paper_APSCON_IEEE_eXpress.pdf
```

---

## Setup Instructions

### Environment Setup (Conda)

```bash
# Create environment from the provided YAML
conda env create -f environment.yml

# Activate environment
conda activate vlm_ftune

# (Optional) Install Jupyter kernel
python -m ipykernel install --user --name=vlm_ftune --display-name="Python (Florence2 Robustness)"
```

**Key dependencies:** PyTorch 2.6.0 + CUDA 11.8, Transformers 4.51.0, pycocotools 2.0.8, Pillow, NumPy, Matplotlib, tqdm.

### Dataset Setup (COCO 2017)

Download the COCO 2017 validation images and annotations. Place them as:

```
./Dataset/
└── coco/
    ├── images/
    │   └── val2017/          # 5,000 validation images
    └── annotations/
        └── annotations/
            └── instances_val2017.json
```

If your dataset is elsewhere, update `ann_file` and `image_dir` paths in the notebook cells.

---

## Usage / Running Experiments

```bash
conda activate vlm_ftune
jupyter lab
```

Open `F2_final_fgsm.ipynb` or `F2_final_pgd.ipynb` and run all cells sequentially. Each notebook:
1. Loads Florence-2-Base and the COCO dataset
2. Runs clean evaluation (baseline mAP)
3. Generates adversarial examples and evaluates (attacked mAP)
4. Applies the defense pipeline and evaluates (defended mAP)
5. Prints the defense effectiveness analysis

**GPU recommended.** Full evaluation takes 6--10+ hours per notebook. For quick testing, add `files = files[:50]` before the main loop.

---

## References

### Core References (This Work)
1. Xiao, B. et al. "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks." CVPR, 2024.
2. Goodfellow, I. et al. "Explaining and Harnessing Adversarial Examples." ICLR, 2015. (FGSM)
3. Madry, A. et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR, 2018. (PGD)

### Key Recent Work (2025--2026)
4. Zhao, J. et al. "One Object, Multiple Lies: Cross-task Adversarial Attack on Unified VLMs." arXiv:2507.07709, 2025.
5. Fu, X. & Zhang, L. "Adversarial Defense in Vision-Language Models: An Overview." arXiv:2601.12443, 2026.
6. Darabi, N. et al. "EigenShield: Causal Subspace Filtering via Random Matrix Theory." arXiv:2502.14976, 2025.
7. Huang, H. et al. "X-Transfer Attacks: Super Transferable Adversarial Attacks on CLIP." ICML, 2025.
8. Wang, Z. et al. "Double Visual Defense." arXiv:2501.09446, 2025.
9. Fox, J. et al. "Adversarial Robustness of Vision in Open Foundation Models." IEEE Access, 2025.
10. Xie et al. "Chain of Attack." CVPR, 2025.
11. Croce, F. & Hein, M. "Reliable evaluation of adversarial robustness." ICML, 2020. (AutoAttack)
12. Carlini, N. & Wagner, D. "Towards Evaluating the Robustness of Neural Networks." IEEE S&P, 2017. (C&W)
13. Tramer, F. et al. "On Adaptive Attacks to Machine Learning Defenses." arXiv:2002.09532, 2020.
14. Nie, W. et al. "Diffusion Models for Adversarial Purification." ICML, 2022.
15. Schlarmann, C. et al. "Robust CLIP: Unsupervised Adversarial Fine-Tuning." ICML, 2024. (FARE)
16. La Torre, A.P. "Adversarial attacks against Modern Vision-Language Models." arXiv:2603.16960, 2026.

---
