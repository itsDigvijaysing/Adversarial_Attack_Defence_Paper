# Robustifying Zero-Shot Detection and OCR Against Adversarial Attacks

This repository studies whether input-transform defenses protect open-vocabulary models from
adversarial attacks. It covers three models, all evaluated on **COCO val2017**:

| Model | Type | Task scored |
|---|---|---|
| YOLOv8x-WorldV2 | one-stage open-vocabulary detector | detection, COCO mAP@[.5:.95] |
| Florence-2-Base | sequence-to-sequence VLM | detection (`<OD>`, mAP) and OCR (self-consistency) |
| SmolVLM-256M-Instruct | chat-style VLM (added in Phase 4) | yes/no object-presence questions, accuracy |

**Attacks:**
- Phase 3: FGSM, PGD and Patch.
- Phase 4: oblivious PGD, and an **adaptive BPDA+EOT PGD** that attacks through the defense.

**Defenses:**
- 13 single transforms: JPEG, Median, Gaussian, TVM, Blur+TVM, Bilateral, SVD, bit depth, dithering, NLM, anisotropic diffusion, BM3D and random resize.
- 7 detection-level ensembles that combine members by class-aware NMS: EnsJMG, EnsJMT, Ens4W, EnsBTC, EnsBMT, EnsJBT and EnsJTS.

**Where to find what:**
- Paper draft: `docs/paper_draft/main.tex`. Its `README.md` traces every number to a source file.
- Short write-up of the Phase-4 results: `docs/PHASE4_RESULTS_SUMMARY.md`.
- Project state and citations: `docs/memory.md`.

## Key findings (18 Nov 2026)

1. **The Phase-3 numbers are correct and reproduce exactly.** We re-ran the clean and oblivious-defended scores in Phase 4:
   - YOLO clean: 0.4519 on 5,000 images and 0.4764 on 1,000.
   - Florence clean: 0.3304 on 5,000 and 0.3604 on 1,000.
   - YOLO oblivious PGD + defense: Median 0.366 (Phase 3: 0.364), EnsJMG 0.407 (Phase 3: 0.405).
2. **Those recoveries only hold against an attacker who ignores the defense.** An adaptive attacker with the same 7/255 budget brings YOLO-World with a defense down to 0.017–0.069 mAP, against 0.476 clean and 0.015 with no defense. On Florence-2 a defense keeps only +0.01 to +0.03 mAP over no defense.
3. **Ensembles are the least bad option, but not a fix.** They keep 3–4 times more mAP than Median under the adaptive attack (0.048–0.069 vs 0.017). He et al. (WOOT 2017) found the same weakness for ensembles of weak defenses on image classifiers.
4. **Ensembles are cheap on clean detection** (≤ 0.010 mAP on 5,000 images) **but expensive on OCR** (self-consistency 0.64–0.84).
5. **SmolVLM, oblivious attack only:** strong smoothing (Blur+TVM, random resize, BM3D) recovers about 70–75% of the damage. The ensembles recover 8–24%.

## Status

**Phase 3: complete.**
- v2 at N=1,000 covers all 9 tracks: 3 attacks × {YOLO detection, Florence detection, Florence OCR}.
- The survey at N=5,000 covers all three tracks. Only the Florence-2 detection survey files are in this repo.
- The N=5,000 YOLO and OCR *attacked* numbers come from the earlier paper's tables. Their summary files no longer exist.
- The N=5,000 YOLO *clean* baseline (0.4519) is now backed by a committed file: `results/phase4/clean_control_yolo/`.

**Phase 4: complete (18 Nov 2026).**
- 20 GPU jobs ran on one RTX 5000 Ada (32 GB) from 17 Nov 03:36 to 18 Nov 03:10, with 0 failures.
- Every job ran with `--deterministic`, seed 1 unless it was a seed repeat, and the first N sorted val2017 images.
- Results: `results/phase4/`. Queue logs (local only): `logs/phase4_queue/`.

**Paper:** the RQ3 section of `docs/paper_draft/main.tex` (Table 5, checks i–v, the clean-cost paragraph) still has red `\pending{}` items. Fill them from the tables below.

**Warnings:**
- `results/results_survey_florence_detection/` (no attack suffix) is a **3-image smoke run**. Its `num_images` is 3 and its clean mAP 0.4998. It is not a result.
- Phase-3 outputs without `_v2` are historical and may include contamination from checkpoint reuse. Do not cite them.

## Results

**Rules for every table below:**
- Every number is read from a committed `summary*.json` unless marked otherwise. The old tables were re-checked against their files on 18 Nov 2026.
- Detection is COCO mAP@[.5:.95].
- OCR is **self-consistency**: how closely the OCR text matches the model's *own* clean output. It is not accuracy.
- **N=1,000 and N=5,000 are different image subsets. Never mix them in one table or one sentence.**

### Phase 4 — adaptive attack on YOLO-World (N=1,000)

**Common settings:**
- PGD with eps 0.03, which is at most 7/255 per pixel at native resolution.
- 50 steps, alpha 0.002, `--attack-space native`, `--pgd-select best`.

**How the oblivious and adaptive attacks differ:**
- The oblivious attack ignores the defense.
- The adaptive attack differentiates through it: BPDA for the non-differentiable parts, the true gradient for Median, and EOT over the ensemble branches.
- Both attacks start from the same seeded point.

**Reference values:**
- Clean, no defense: **0.476**.
- PGD, no defense: **0.015** (3-seed mean; seed 1 alone gives 0.018).

| Defense | Clean + defense | Oblivious PGD + defense | Adaptive, max objective | Adaptive, mean objective | Worst case |
|---|---|---|---|---|---|
| Median | 0.458 | 0.366 | 0.018 (3 seeds, sd 0.001) | same (one branch) | **0.017** |
| EnsJMG (JPEG + Median + Gaussian) | 0.466 | 0.407 | 0.073 (3 seeds, sd 0.002) | 0.048 | **0.048** |
| Ens4W (JPEG + Median + TVM + Gaussian) | 0.468 | 0.431 | 0.108 | 0.069 | **0.069** |

Notes:
- The mean-objective runs, the worst-case column and every Ens4W number are seed 1 only.
- Per seed, adaptive max: Median 0.0165 / 0.0181 / 0.0191; EnsJMG 0.0713 / 0.0746 / 0.0741.
- Source: `results/phase4/adaptive_bpda_yolo/summary_adaptive_bpda_yolo_<target>_seed<S>[_eot-loss-mean]_attack-space-native_pgd-select-best_latest.json`.

### Phase 4 — adaptive attack on Florence-2 detection (N=1,000, seed 1)

The attack works in pixel space at 7/255. It uses token cross-entropy against the model's own clean decode, with the same PGD loop and surrogates as YOLO.

**Reference values:**
- Clean, no defense: **0.360**.
- Oblivious PGD, no defense: **0.134–0.138**. Identical runs differ because Florence is not bit-repeatable.

| Defense | Clean + defense | Oblivious PGD + defense | Adaptive PGD + defense |
|---|---|---|---|
| Median | 0.350 | 0.256 | **0.148** |
| EnsJMG | 0.353 | 0.264 | **0.166** |

Sources:
- `results/phase4/adaptive_bpda_florence/summary_adaptive_bpda_florence_<target>_seed1_pgd-select-best_latest.json`.
- Mean vs max objective on EnsJMG, first 300 images (0.201 vs 0.199): `results/phase4/adaptive_bpda_florence_n300/`.

### Phase 4 — checks that the attack is sound

| Check | Setting | Result | Meaning |
|---|---|---|---|
| Adaptive vs oblivious objective | every 1,000-image run | the adaptive objective is lower on 999–1,000 of 1,000 images | the optimizer does what it should |
| More steps | YOLO EnsJMG, 300 images | 50 steps 0.086 → 100 steps 0.054 | not converged at 50 steps; Table 5 is an **upper bound** |
| Larger budget | YOLO EnsJMG, eps 0.1, 300 images | 0.023 | mAP keeps falling as eps grows; no sign of gradient masking |
| Max vs mean objective | YOLO EnsJMG and Ens4W; Florence EnsJMG | YOLO: mean is stronger (0.071 → 0.048, 0.108 → 0.069). Florence: equal (0.199 vs 0.201) | report the **worst case over objectives** |
| Median gradient | YOLO Median, true vs identity (BPDA) | 0.017 vs 0.042 | the true median gradient is the stronger attack |
| Repeatability | `--deterministic`, same seed, run twice | YOLO bit-identical; Florence varies about 0.005 mAP | state repeatability per model |

Sources: `results/phase4/adaptive_bpda_yolo_iters50_n300/`, `adaptive_bpda_yolo_iters100_n300/`, `adaptive_bpda_yolo_eps0.1/`, and the `_median-surrogate-identity` and `_eot-loss-mean` summaries in `adaptive_bpda_yolo/`.

**Note on the 300-image runs:** the first 300 images of a 1,000-image run are the same images. `scripts/p4_subset_score.py` scores them from the 1,000-image checkpoint. On YOLO it reproduces a separate 300-image run exactly.

### Phase 4 — clean cost of every defense (N=5,000, no attack)

**Baselines with no defense:** YOLO 0.4519 mAP, Florence detection 0.3304 mAP, OCR 1.000 (by construction).

| Defense | YOLO mAP change | Florence det. mAP change | OCR self-consistency |
|---|---|---|---|
| EnsJTS | +0.000 | −0.007 | 0.843 |
| Ens4W | −0.004 | −0.009 | 0.746 |
| EnsJBT | −0.004 | −0.005 | 0.789 |
| EnsBMT | −0.004 | −0.005 | 0.788 |
| EnsJMG | −0.005 | −0.006 | 0.713 |
| EnsJMT | −0.005 | −0.006 | 0.746 |
| EnsBTC | −0.010 | −0.010 | 0.644 |
| SVD (keeps 90% of the singular values by count; close to identity) | 0.000 | −0.002 | 0.953 |
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

Sources: `results/phase4/clean_control_{yolo,florence_det,florence_ocr}/summary_clean_control_*_alltransforms_latest.json`.

The OCR column bounds OCR recovery: a transform cannot restore more self-consistency under attack than it keeps on clean images.

### Phase 4 — SmolVLM-256M yes/no presence questions (N=1,000, 2,000 questions)

**Setup:**
- Each image is resized to a 512×512 square in every condition.
- Questions are "Is there a {class} in this image?", half yes and half no.
- Attack: oblivious PGD, eps 0.03, 10 steps, alpha 0.0075.
- An ensemble answers with the mean of its members' yes-probabilities.

**Reference values:** clean accuracy **0.856**, attacked accuracy **0.439**.

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

Source: `results/phase4/presence_qa_smolvlm/summary_presence_qa_smolvlm_latest.json`. That file also has bootstrap intervals for every clean cost and recovery.

We did not run an adaptive attack on SmolVLM. Given the YOLO and Florence results, read these recoveries as upper bounds.

### How Phase 4 changes the Phase-3 conclusions

These are YOLO-World values on the same 1,000 images.

| | Phase 3 | Phase 4, oblivious | Phase 4, adaptive (worst case) |
|---|---|---|---|
| Attack | PGD, 10 steps, alpha 0.0075, letterbox space | PGD, 50 steps, alpha 0.002, native resolution | BPDA+EOT through the defense, 50 steps |
| No defense | 0.0737 | 0.015 | 0.015 |
| Median | 0.3639 | 0.366 | 0.017 |
| EnsJMG | 0.4045 | 0.407 | 0.048 |
| Best Phase-3 defense (EnsBTC) | 0.4128 | not run | not run |

- **The defended numbers reproduce Phase 3**, so the code and data agree with the paper. The stronger Phase-4 PGD hurts the undefended model more (0.015 vs 0.074). That makes the defenses look slightly *better* under the Phase-3 threat model.
- **The change is the threat model.** Once the attacker knows the defense, the Phase-3 recovery (+0.29 to +0.33 mAP) disappears. That includes the ensembles the Phase-3 paper recommended.
- **What the paper must change:**
  - Present the ensemble result two-sided: it helps against an unaware attacker, and little survives an aware one.
  - Present Table 5 as an upper bound.
  - Report the worst case over the objectives.
  - Drop the claim that the "max" objective is the natural worst case, and the "union-aware" novelty claim. On YOLO the mean objective was stronger.
- **Florence Phase-3 PGD numbers are not comparable to Phase 4.** Phase 3 attacked Florence in normalized space (about 1.7/255 per pixel); Phase 4 attacks it at 7/255 in pixel space.

### Phase 3 — Locked v2, N=1,000 (the earlier paper's main table)

The scripts in the *generated by* column now live in `archive/Phase3/`. Every row was re-checked against its `summary.json` on 18 Nov 2026.

| Track | clean | attacked | best defense | defended | generated by | source file |
|---|---|---|---|---|---|---|
| YOLO · FGSM | 0.4765 | 0.2259 | blur_tvm | 0.3448 (+47.4%) | `FGSM_Phase3_YOLO_v2.ipynb` | `results/results_phase3_yolo_fgsm_v2/summary.json` |
| YOLO · PGD (10 iters) | 0.4765 | 0.0737 | ens_blur_tvm_combo | 0.4128 (+84.2%) | `PGD_Phase3_YOLO_v2.ipynb` | `results/results_phase3_yolo_pgd_v2/summary.json` |
| YOLO · Patch | 0.4765 | 0.4537 | ens_jpeg_median_tvm | 0.4599 (+27.5%) | `Patch_Phase3_YOLO_v2.ipynb` | `results/results_phase3_yolo_patch_v2/summary.json` |
| Florence · FGSM | 0.3605 | 0.2731 | ens_jpeg_median_gaussian | 0.3281 (+62.9%) | `FGSM_Phase3_Florence_v2.py` | `results/results_phase3_florence_fgsm_v2/summary.json` |
| Florence · PGD | 0.3605 | 0.2243 | ens_jpeg_median_gaussian | 0.3245 (+73.5%) | `PGD_Phase3_Florence_v2.py` | `results/results_phase3_florence_pgd_v2/summary.json` |
| Florence · Patch | 0.3605 | 0.3268 | ens_blur_tvm_combo | 0.3324 (+16.7%) | `Patch_Phase3_Florence_v2.py` | `results/results_phase3_florence_patch_v2/summary.json` |
| OCR · FGSM | 1.0000 | 0.3287 | tvm | 0.5615 (+34.7%) | `FGSM_Florence2_OCR_Robust.py` | `results/results_fgsm_florence2_ocr_robust/summary.json` |
| OCR · PGD (alpha 0.003) | 1.0000 | 0.3139 | tvm | 0.6384 (+47.3%) | `PGD_Florence2_OCR_Robust.py` | `results/results_pgd_florence2_ocr_robust/summary.json` |
| OCR · Patch | 1.0000 | 0.3675 | median | 0.5566 (+29.9%) | `Patch_Florence2_OCR_Robust.py` | `results/results_patch_florence2_ocr_robust/summary.json` |

All defenses under YOLO PGD, from `results/results_phase3_yolo_pgd_v2/summary.json`:

| Defense | mAP |
|---|---|
| EnsBTC | 0.4128 |
| EnsJMT | 0.4114 |
| Gaussian | 0.4053 |
| TVM | 0.4049 |
| EnsJMG | 0.4045 |
| Blur+TVM | 0.4025 |
| JPEG | 0.3643 |
| Median | 0.3639 |

> The N=1,000 clean baselines (**YOLO 0.4765**, **Florence 0.3605**) belong to the v2 subset. They are
> *not* the survey's N=5,000 clean baselines (Florence-det **0.3300**, YOLO **0.4519**). The OCR
> `clean` column is 1.0 **by construction**.

> **Correction (kept):** earlier notes quoted YOLO PGD as `clean 0.4972 → 0.1425 → 0.4801` and a
> "stronger PGD" `0.4765 → 0.0614 → 0.4263`. Those match **no committed v2 result**; they trace to a
> 100-image scratch run. Cite only committed `summary*.json` values.

### Phase 3 — Survey, N=5,000

The Florence-2 detection rows are read from committed files produced by
`archive/Phase3/run_survey_florence_detection.py --tier survey --num-images 5000`. That run has 19 defense rows (13 solos and 6 ensembles), `novel: false` and 0 failed images. Re-checked on 18 Nov 2026.

| Track | clean | attacked | best defense | defended | source file |
|---|---|---|---|---|---|
| Florence-det · FGSM | 0.3300 | 0.2488 | ens_jpeg_bilateral_tvm | 0.2942 (+55.9%) | `results/results_survey_florence_detection_fgsm/summary_fgsm.json` |
| Florence-det · PGD | 0.3300 | 0.2005 | ens_jpeg_median_tvm | 0.2961 (+73.8%) | `results/results_survey_florence_detection_pgd/summary_pgd.json` |
| Florence-det · Patch | 0.3300 | 0.2958 | bilateral | 0.3165 (+60.6%) | `results/results_survey_florence_detection_patch/summary_patch.json` |

The YOLO and OCR rows below are from the earlier paper's tables (Table 3 and Table 5). The runs were real, but **their summary files no longer exist**. The YOLO clean value is now also backed by the committed Phase-4 file.

| Track | clean | FGSM | PGD | Patch | source |
|---|---|---|---|---|---|
| YOLO det (mAP) | 0.4519 | 0.2051 | 0.0619 | 0.4257 | earlier paper Table 3; clean reproduced in `results/phase4/clean_control_yolo/` |
| Florence-2 OCR (self-consistency) | 1.000 | 0.346 | 0.294 | 0.383 | earlier paper Table 5 |

- `bilateral` (a Tier-2 addition, absent from the locked v2 set) is the strongest **solo** on all three Florence-det attacks at N=5,000, and wins Patch outright. The N=1,000 v2 ranking does not carry over.
- `random_resize` is the worst row on all three attacks (e.g. Patch 0.1449, −0.1510). This is a documented negative result, not a bug.
- The N=5,000 PGD alpha is eps/4 on all three tracks. Never mix in the N=1,000 OCR PGD row (alpha 0.003).

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
│   └── p4_logging.py           # run-artifact policy (run.log, ephemeral eval dumps)
├── scripts/
│   ├── setup_server.sh         # fresh server: conda env, COCO, weights, preflight, tests
│   ├── p4_launch.sh            # starts the Phase-4 plan in a detached tmux session
│   ├── p4_queue.sh             # runs one queue of commands with logs, done markers, one retry
│   ├── p4_status.sh · p4_health.py   # read-only progress and health checks
│   └── p4_subset_score.py      # scores the first K images of a finished adaptive checkpoint
├── tests/                      # Phase-4 checks; none need a GPU, weights or COCO
├── results/
│   ├── phase4/<experiment>_<model>/   # Phase-4 runs (timestamped summaries + run logs)
│   └── results_*/              # Phase-3 outputs (summary*.json, run.log, .png tracked)
├── docs/
│   ├── paper_draft/            # main.tex, refs.bib, table/figure generators, Overleaf zip
│   ├── PHASE4_RESULTS_SUMMARY.md · memory.md · HANDOFF_A5000.md
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

# Adaptive vs oblivious, YOLO-World, 1,000 images (Table 5)
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

# Checks that need no GPU
python phase4/p4_yolo.py --preflight
python tests/test_offline_logic.py      # 38 torch-free checks
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

These are seconds per image as the runs reported them. Up to 8 jobs shared the one GPU, so a job running alone is faster. Peak GPU memory with 8 jobs was 30.4 of 32.7 GB.

| Run | s/image | images |
|---|---|---|
| YOLO `clean_control --all-transforms` | 1.8 | 5,000 |
| YOLO adaptive, Median | 10–12 | 1,000 |
| YOLO adaptive, EnsJMG | 15–17 | 1,000 |
| YOLO adaptive, Ens4W (max / mean objective) | 26.5 / 13.7 | 1,000 |
| Florence adaptive, Median / EnsJMG | 45.5 / 67.6 | 1,000 |
| Florence `clean_control`, detection / OCR | 16.9 / 14.8 | 5,000 |
| SmolVLM presence QA | 26.7 | 1,000 |

`--deterministic` costs about 19% on YOLO. It makes YOLO runs bit-identical, but not Florence runs.

### The experiments

| Experiment | Model(s) | What it answers |
|---|---|---|
| `clean_control` | YOLO, Florence | Do these transforms counter *attacks*, or do they just help or hurt images in general? Scores every transform on clean images. `--all-transforms` adds bit depth, dithering, NLM and anisotropic diffusion (OCR also BM3D and random resize) under its own `_alltransforms` summary name. |
| `adaptive_bpda` | YOLO, Florence `<OD>` | What happens when the attacker knows the defense? PGD runs through the `--target` defense with BPDA and EOT, and is compared with the same PGD run obliviously from the **same seeded start**. Reports clean / oblivious / adaptive mAP with and without the defense. |
| `presence_qa` (`p4_smolvlm.py`) | SmolVLM-256M | Do the transforms protect a chat-style VLM? Yes/no questions built from the COCO annotations, oblivious PGD only. |

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
