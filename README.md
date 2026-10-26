# Robustifying Zero-Shot Detection and OCR Against Adversarial Attacks

This repository studies adversarial robustness for two open-vocabulary models:
- Florence-2-Base (sequence-to-sequence VLM)
- YOLOv8x-WorldV2 (one-stage detector)

Attacks: FGSM, PGD, Patch

Defenses: solo transforms + detection-level transform ensembles merged by class-aware NMS.

Primary report: `docs/CS24MTECH14020_CVPR_Project_Report.pdf`

## Current Status (Important)

- **Phase 3 v2 at N=1000 is complete for all 9 tracks** (3 attacks × {YOLO-det, Florence-det,
  Florence-OCR}). Use the `*_v2` notebooks/`*_Robust.py` scripts and their `summary.json`.
- **Survey N=5000 has been run for all three tracks, but only Florence-2 detection is committed
  here.** Each track ran on its owner's own GPU (see the team table below): Florence-2 detection
  by Lokendra — its three `summary_{attack}.json` live under
  `results/results_survey_florence_detection_{fgsm,pgd,patch}/` — while the **YOLO** (Digvijay) and
  **OCR** (Ankush) N=5000 runs completed on their machines and their numbers are confirmed, but the
  output directories have not been copied into this repo yet. **Action: import
  `results_survey_yolo/summary_{attack}.json` and `results_survey_florence_ocr/summary_{attack}.json`
  from their machines and commit them**, so every paper number resolves to a file here. Until then
  the N=5000 YOLO/OCR figures are valid results with an off-repo source, not repo-citable ones.
- ⚠️ `results/results_survey_florence_detection/` (no attack suffix) holds a **3-image smoke run**
  (`num_images: 3`, clean_mAP 0.4998). It is not a result. The N=5000 numbers live in the
  `_fgsm` / `_pgd` / `_patch` suffixed directories.
- Older Phase 3 outputs without `_v2` are historical and may include contamination from checkpoint
  reuse — do **not** cite them. The survey scripts use a config-keyed `SurveyCheckpoint` that cannot
  reproduce that contamination.
- **Phase 4 has started.** All Phase-3 code moved to `archive/Phase3/` (read-only — see
  `archive/Phase3/README.md`). Active work now runs through **two** entry points, one per model:
  `phase4/p4_yolo.py` and `phase4/p4_florence.py`.

## Team

| Member | Track | Phase-4 command |
|---|---|---|
| Ankush | Florence-2 OCR | `python phase4/p4_florence.py --tracks florence_ocr` |
| Digvijay | YOLOv8x-worldv2 detection | `python phase4/p4_yolo.py` |
| Lokendra | Florence-2 detection | `python phase4/p4_florence.py --tracks florence_det` |

Each member runs independently on an RTX GPU (`conda activate vlm_ftune`), one process per GPU,
one tmux pane per process.

## Repository Structure

```
.
├── phase3_common.py            # shared core: defenses, ensembles, NMS merge, eval, checkpoint.
│                               # STAYS AT THE ROOT — phase4/ imports it, and importing the SAME
│                               # module is what makes Phase-4 numbers comparable to Phase 3.
├── phase4/
│   ├── p4_core.py              # everything not model-specific, written ONCE: CLI, image
│   │                           # selection, defense branches, checkpointed loop, COCO/OCR
│   │                           # scoring, reports, the clean_control experiment
│   ├── p4_yolo.py              # ENTRY POINT — YOLOv8x-worldv2 (+ the BPDA/EOT adaptive attack)
│   ├── p4_florence.py          # ENTRY POINT — Florence-2 detection + OCR from one model load
│   └── p4_logging.py           # run-artifact policy (run.log, ephemeral eval dumps)
├── tests/                      # Phase-4 checks, none need a GPU, weights or COCO:
│                               #   test_offline_logic.py (torch-free), test_flow_stubbed.py
│                               #   (drivers, stubbed model), test_attack_units.py (CPU torch)
├── results/
│   ├── phase4/<experiment>_<model>/   # Phase-4 runs (timestamped summaries + run.log)
│   └── results_*/              # Phase-3 outputs (summary*.json, run.log, .png tracked)
├── README.md · CLAUDE.md · environment.yml · .gitignore
├── docs/         # report + Presentation PDFs, memory.md, memory_phase2_archive.md, GPU, tmux.txt
├── figures/      # Final_Results_Images/, Result_Images/
└── archive/      # read-only history
    ├── Phase3/   # the 9 locked v2 scripts/notebooks + the 3 N=5000 survey scripts + README
    ├── Scripts_Extra/ · Logs_Extra/ · Very_OLD/
```

Data/model assets used by runs (gitignored, fetched separately): `val2017/`,
`annotations/instances_val2017.json`, `yolov8x-worldv2.pt` (auto-downloaded by ultralytics).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate vlm_ftune
python -m ipykernel install --user --name=vlm_ftune --display-name "Python (vlm_ftune)"
```

## Dataset Layout

Current notebooks expect COCO validation data at repository root:

```text
./val2017/
./annotations/instances_val2017.json
```

The Phase-4 entry points auto-resolve both paths (they try `./val2017`, `./Dataset/val2017`, ...),
so no edit is needed; pass `--image-dir` / `--ann-file` if your layout differs. The archived
Phase-3 scripts have hardcoded, per-track defaults — see `archive/Phase3/README.md`.

## Phase 4 — how to run

Two entry points, one per model. Each is one process on one GPU, i.e. one tmux pane. `--preflight`
needs no GPU and lists anything missing; `--smoke-test` runs 2 images (CPU unless `--device` is
given) into a `smoke_test/` subdirectory that is gitignored and stamped `"smoke_test": true`.
**A real run stops with an error if the image directory holds fewer than `--n` images** (the
message names http://images.cocodataset.org/zips/val2017.zip); only `--smoke-test` tolerates it.
`--preflight` counts the images the same way and says NOT READY (`SHORT`) in that case.

```bash
conda activate vlm_ftune

# --- clean-image control: every defense on UNATTACKED images -----------------
python phase4/p4_yolo.py     --experiment clean_control --n 1000 --gpu 0
python phase4/p4_florence.py --experiment clean_control --n 1000 --gpu 0   # both tracks
python phase4/p4_florence.py --experiment clean_control --tracks florence_ocr --gpu 1

# --- adaptive white-box: BPDA + EOT PGD through a defense (YOLO only) -------
# vs EnsJMG (default target), n=1000, 3 seeds (seed = PGD init draw; same images)
for s in 0 1 2; do
  python phase4/p4_yolo.py --experiment adaptive_bpda --target ens_jpeg_median_gaussian \
      --n 1000 --seed $s --gpu 0
done
python phase4/p4_yolo.py --experiment adaptive_bpda --target median   --n 1000 --seed 0 --gpu 0
python phase4/p4_yolo.py --experiment adaptive_bpda --target ens_4way --n 1000 --seed 0 --gpu 0

# convergence check (50 iterations) and large-eps sanity check, own dirs
python phase4/p4_yolo.py --experiment adaptive_bpda --n 300 --pgd-iters 50 --seed 0 --gpu 0 \
    --output-dir results/phase4/adaptive_bpda_yolo_iters50
python phase4/p4_yolo.py --experiment adaptive_bpda --n 100 --eps 0.1 --seed 0 --gpu 0 \
    --output-dir results/phase4/adaptive_bpda_yolo_eps0.1

# --- checks that need no GPU -------------------------------------------------
python phase4/p4_yolo.py --preflight
python phase4/p4_florence.py --experiment clean_control --smoke-test
python tests/test_offline_logic.py      # 37 torch-free checks
python tests/test_flow_stubbed.py       # 20 driver checks, model + attacks stubbed
CUDA_VISIBLE_DEVICES='' python tests/test_attack_units.py   # 14 CPU-torch attack checks
```

Shared flags: `--n/--num-images`, `--select {head,random}`, `--seed`, `--tracks`, `--gpu`,
`--device {auto,cuda,cpu}`, `--checkpoint-every`, `--no-checkpoint`, `--keep-eval-dumps`,
`--no-per-image`, `--log-every N`, `--verbose-eval`, `--skip-clean-consistency`, plus the defense hyperparameters (defaults match Phase 3 exactly:
JPEG q75, median 3x3, TVM w=0.05/200 iters, sigma=1.0, SVD 90%, NMS IoU 0.5).

`--select head` (the default) picks the first `--n` of the sorted file list — exactly the subset the
Phase-3 attacked runs scored — so the deltas are same-subset comparable. The chosen image ids and a
SHA1 digest of them go into every summary.

### Measured throughput (RTX 3060 Laptop, 6 GB, verified 2026-08-24)

Real `s/image` from GPU runs on this hardware — not estimates. All exited 0.

| Run | s/image | n | n=1000 | n=5000 | peak VRAM |
|---|---|---|---|---|---|
| `p4_yolo.py --experiment clean_control` | **0.67** | 10 | 0.19 h | 0.93 h | ~2.3 GB |
| `p4_yolo.py --experiment adaptive_bpda` | 1.61 | 2 | 0.45 h | 2.24 h | ~2.3 GB |
| `p4_florence.py` `florence_det` (5 beams) | 2.44 | 2 | 0.68 h | 3.39 h | ~1.3 GB |
| `p4_florence.py` `florence_ocr` (5 beams) | 1.13 | 2 | 0.31 h | 1.57 h | (same load) |

Only the YOLO `clean_control` row is measured over 10 images; the rest are 2-image runs, so treat
them as ±30% and trust the `s/image` each run prints for itself. The `adaptive_bpda` row predates
the current attack (it ran 2 PGD iterations with the `mean` EOT loss); the defaults now run 10
iterations with the `max` loss, which adds one no-grad forward per branch per step, so expect it
to be slower per image (5x the PGD steps, plus those forwards) until re-measured. The Florence rows use the
production decode (`--num-beams 5`), not the reduced `--smoke-test` decode.

6 GB is enough for either model at batch 1 (YOLO peaked ~2.3 GB, Florence ~1.3 GB of 5.8 GB free,
and VRAM returned to idle after each run). **Do not run both entry points on the same 6 GB card at
once** — run them in sequence. On a larger card they can share.

### The two experiments

| Experiment | Model(s) | What it answers |
|---|---|---|
| `clean_control` | both | Do these transforms counter *attacks*, or do they just help/hurt images in general? Scores the 7 ensembles (EnsJMG, EnsJMT, Ens4W, EnsBTC, EnsBMT, EnsJBT, EnsJTS) and 6 solos (JPEG, Median, Gaussian, TVM, Blur+TVM, Bilateral — plus SVD, needed as an EnsJTS member) on clean images. **A small negative delta is an expected, publishable result**: it is the clean-accuracy cost of the transform. |
| `adaptive_bpda` | YOLO only | What happens when the attacker knows the defense? PGD eps=0.03/10 iters/alpha=eps/4 through the `--target` defense with BPDA and EOT, against the same PGD run obliviously from the **same seeded start**. Reports clean / oblivious-defended / adaptive-defended mAP on the same subset, plus the attacker's own objective J through the target. Its approximations, for the options chosen, are in the summary's `implementation_uncertainty` field and the `.md`. |

Adaptive-attack flags (each enters the checkpoint signature, the summary JSON `attack_flags` block
and the `.md` header):

| Flag | Default | Meaning |
|---|---|---|
| `--target` | `ens_jpeg_median_gaussian` (EnsJMG) | also `ens_4way` (Ens4W), `ens_jpeg_median_tvm` (EnsJMT), `ens_blur_tvm_combo` (EnsBTC), solos `median`, `jpeg`, `gaussian`, `tvm`, `blur_tvm`. Ensemble members are read from `phase3_common.TIER1_ENSEMBLES`; the scored defense is always phase3_common's |
| `--eot-loss` | `max` | `max` = union-aware: per anchor, suppress the most confident branch (the NMS union keeps a box if ANY branch does). `mean` = mean of the branch objectives (the earlier pilot) |
| `--restarts` | `1` | seeded restarts per image (both attacks); keeps the lowest surrogate objective |
| `--median-surrogate` | `true` | `true` = exact almost-everywhere gradient of the unfold median; `identity` = BPDA straight-through. JPEG and TVM are always identity, Gaussian always its true gradient |
| `--quantize` | `truncate` | final uint8 conversion; `truncate` is Phase 3's, `round` rounds |
| `--attack-space` | `letterbox` | `letterbox` (Phase 3) optimises the square 640 letterbox, so the surrogate defense runs at a different scale than the deployed one. `native` optimises at native resolution and feeds the model through a differentiable copy of ultralytics' predict letterbox |

`--seed` sets the per-(seed, image id, restart) PGD init draw, shared by the oblivious and the
adaptive attack; with `--select head` it does not change the images. Adaptive summaries are named
`summary_adaptive_bpda_yolo_<target>_seed<S>[_<flag>-<value>...]_n<N>_<timestamp>.json` (+ `_latest`),
where each non-default flag of the table above except `--target` is appended (e.g.
`_median-surrogate-identity`), so every target/seed/ablation keeps its own citable `_latest` in the
one resumable run directory. `--eps` / `--pgd-iters` diagnostics still need their own `--output-dir`.
In `native` space the uint8 attack output is clipped to the clean image +/- floor(eps*255)/255
(7/255 at eps 0.03, since 0.03*255 = 7.65); `letterbox` keeps Phase 3's truncation.

Both reuse `phase3_common.apply_all_defenses_gpu` and the same class-aware NMS merge
(`assemble_results` / `merge_branches_nms`, IoU 0.5) as the Phase-3 attacked runs — that identity is
the only reason the numbers can be compared.

### Phase 3 (archived)

The nine locked v2 scripts/notebooks and the three N=5000 survey scripts now live in
`archive/Phase3/`. **Read `archive/Phase3/README.md` before touching them** — they need
`PYTHONPATH=.` from the repo root, they write to old root paths, their dataset paths differ by
track, and their PGD alpha is not uniform:

> ⚠️ **PGD step size is NOT uniform across tracks.** `archive/Phase3/PGD_Florence2_OCR_Robust.py:642`
> defaults to `--alpha 0.003` (= eps/10), and that is the value that produced the committed N=1000
> OCR PGD number (`results/results_pgd_florence2_ocr_robust/summary.json` records `"alpha": 0.003`;
> the run log confirms it). Every other PGD track uses **alpha = eps/4 = 0.0075**:
> `PGD_Phase3_YOLO_v2.ipynb`, `PGD_Phase3_Florence_v2.py:82`,
> `run_survey_florence_detection.py:167`, `run_survey_yolo.py:161`,
> `run_survey_florence_ocr.py:566`. Consequences: (a) a blanket "PGD used alpha = eps/4" claim is
> wrong for the committed N=1000 OCR row; (b) that number and any future N=5000 OCR PGD number are
> **not** step-size comparable. Do not "fix" the 0.003 — the committed result depends on it.

## Run Artifacts and Log Policy

**Every new run produces exactly three shareable files, and nothing large.** Enforced by
`phase4/p4_logging.py`, which both Phase-4 entry points use:

| File | Size | Git | What it is |
|---|---|---|---|
| `summary_*.json` | KB | **committed** | the numbers, the config, the chosen image ids |
| `summary_*.md` | KB | **committed** | the readable table |
| `run.log` | tens of KB | **committed** | full console transcript, one timestamped block per run (appends on resume; records host, git commit, exact command, and the exit status) |
| `per_image_*.json` | ~1 MB at n=5000 | ignored | per-image audit trail — `--no-per-image` skips it |
| `checkpoint_*.pkl` | large | ignored | config-keyed resume state |
| per-condition COCO dumps | **hundreds of MB** | **never written** | pycocotools needs them on disk, so they go to a temp dir that is deleted when the run ends. `--keep-eval-dumps` puts them in `<output-dir>/eval_dumps/` for debugging only |

Each run ends with a **RUN ARTIFACTS** table listing every file it wrote, split into
*committed (share these)* and *local only (do not share)*, and warns if any committed file exceeds
5 MB. So `git add results/` is safe and a run directory can be zipped and sent to a teammate as-is.

### What the log deliberately does NOT contain

Measured on `logs/survey_fgsm_n5000.log` (a Phase-3 run): **803 of its 1,155 lines — 70% — are
pycocotools boilerplate** (480 `Average Precision/Recall` rows, 200 `Running per image
evaluation…/DONE (t=…)` lines, 123 index-loading lines), of which we use exactly two numbers per
condition. Phase 4 removes three sources of noise:

| Removed | Was | Now |
|---|---|---|
| COCO-eval chatter | ~18 lines × every condition | one `[eval]` line per condition (mAP, AP50, #dets). `--verbose-eval` restores the raw output |
| Progress heartbeat | every 25 images (200 lines/track at n=5000) | every 100 images, `--log-every N` |
| The markdown table | printed in full into `run.log` **and** written to the `.md` | a short RESULT DIGEST (baseline, best 3, worst 1) + the `.md` path |

Estimated effect for an n=5000 two-track run: roughly **750 → 170 lines (~50 KB → ~12 KB)** per
`run.log`, with no loss of anything you would cite. The historical `logs/*.log` from Phase 3 are
left untouched on purpose — they are the evidence behind committed numbers, and 400 KB total is not
a problem worth rewriting history for.

The archived Phase-3 scripts in `archive/Phase3/` do **not** use this — they still write the
per-condition dumps into their output directory (this is where the ~702 MB on disk came from).
`.gitignore` keeps them out of git, but delete `results_*/[!s]*.json` by hand before sharing one of
those directories.

Phase-4 summaries are **timestamped and never overwritten**:
`summary_<experiment>_<model>_n<N>_<YYYYmmdd-HHMMSS>.json`, plus a `_latest.json`/`_latest.md`
mirror that is stable to cite. The run *directory* stays fixed
(`results/phase4/<experiment>_<model>/`) so a crashed run resumes from its checkpoint instead of
restarting from zero — a fresh directory per launch would throw that away. The checkpoint filename
carries the config signature (`checkpoint_<track>_<sig8>.pkl`), so different configurations coexist
instead of invalidating each other.

`.gitignore` matches **files, not directories** (`results/**/*.json` + `!results/**/summary*.json`
+ `!results/**/run.log`); never write `results/*`, because excluding a directory makes every `!`
negation inside it unrescuable and silently drops all committed summaries. `--smoke-test` output
(`results/**/smoke_test/`) is ignored wholesale so a plumbing check can never be mistaken for a
result. For a future public release, large blobs can be purged from history with **BFG Repo
Cleaner** (deferred; all team members must re-clone afterward — see `docs/memory.md`).

## Result Artifacts and Versioning Notes

- `.pkl` outputs are large runtime artifacts and are typically not tracked.
- JSON summaries and comparison outputs are the primary files for analysis/reporting.
- For writing the paper/report, use v2 outputs and cross-check with:
  - `docs/CS24MTECH14020_CVPR_Project_Report.pdf`
  - `docs/memory.md`

## Confirmed Results (every number below is read from a committed `summary*.json`)

Detection = COCO mAP@[.5:.95]; OCR = self-consistency similarity (not accuracy).
**Two different N are in play — never mix them in one table or one sentence.**

### Survey, N=5000 — `run_survey_*.py`

All three tracks have been run at N=5000, one per team member's GPU. Only the Florence-2
**detection** rows below are read from files committed in this repo; they were produced by
`archive/Phase3/run_survey_florence_detection.py --tier survey --num-images 5000` (19 defense rows: 13 solos +
6 ensembles; `novel: false`; 0 failed images).

| Track | clean | attacked | best defense | defended | source file |
|---|---|---|---|---|---|
| Florence-det · FGSM | 0.3300 | 0.2488 | ens_jpeg_bilateral_tvm | 0.2942 (+55.9%) | `results/results_survey_florence_detection_fgsm/summary_fgsm.json` |
| Florence-det · PGD | 0.3300 | 0.2005 | ens_jpeg_median_tvm | 0.2961 (+73.8%) | `results/results_survey_florence_detection_pgd/summary_pgd.json` |
| Florence-det · Patch | 0.3300 | 0.2958 | bilateral | 0.3165 (+60.6%) | `results/results_survey_florence_detection_patch/summary_patch.json` |

- `bilateral` (a Tier-2 addition, absent from the locked v2 set) is the strongest **solo** on all
  three attacks at N=5000 and wins Patch outright — the N=1000 v2 ranking does not carry over.
- `random_resize` is the worst row on all three attacks (e.g. Patch 0.1449, −0.1510) — a
  documented negative result, not a bug.
- **YOLO N=5000 (Digvijay) and OCR N=5000 (Ankush) — run and confirmed, files not yet in this
  repo.** `results_survey_yolo/` and `results_survey_florence_ocr/` do not exist here, so figures
  such as the YOLO clean mAP of 0.4519 at N=5000 are correct measured results whose `summary.json`
  currently lives only on the machine that produced it. Copy those directories in and commit the
  `summary_{attack}.json` so the whole paper table resolves to files in this repo — that is the one
  gap a reviewer asking for artefacts would hit.

### Locked v2, N=1000 — the paper's main table

All the scripts in the *generated by* column now live in `archive/Phase3/`.

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

> The N=1000 clean baselines (**YOLO 0.4765**, **Florence 0.3605**) belong to the v2 track and are
> *not* the survey's N=5000 clean baselines (Florence-det **0.3300** from a committed file; YOLO
> **0.4519** from Digvijay's run, file not yet imported). A table that mixes N=1000 and N=5000 rows
> is wrong. The OCR `clean` column is 1.0 **by construction** (the metric compares each OCR string
> to the model's own clean output).

> **Correction (kept):** earlier notes quoted YOLO PGD as `clean 0.4972 → 0.1425 → 0.4801` and a
> "stronger PGD" `0.4765 → 0.0614 → 0.4263`. Those match **no committed v2 result** (they trace to a
> 100-image scratch run). Cite only committed `summary*.json` values.

Interpretation rule used in this repo:
- Prioritize **recovery from attacked mAP** over naive comparison to clean baseline.

## Citation

If you use this repository, cite the project report in:
- `docs/CS24MTECH14020_CVPR_Project_Report.pdf`

And keep the model/data citations used in the report (Florence-2, YOLO-World, COCO, FGSM, PGD, Patch attack references).