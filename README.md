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
├── tests/test_offline_logic.py # torch-free checks of the Phase-4 glue (no GPU needed)
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
needs no GPU and lists anything missing; `--smoke-test` runs 2 images on CPU into a `smoke_test/`
subdirectory that is gitignored and stamped `"smoke_test": true`.

```bash
conda activate vlm_ftune

# --- clean-image control: every defense on UNATTACKED images -----------------
python phase4/p4_yolo.py     --experiment clean_control --n 1000 --gpu 0
python phase4/p4_florence.py --experiment clean_control --n 1000 --gpu 0   # both tracks
python phase4/p4_florence.py --experiment clean_control --tracks florence_ocr --gpu 1

# --- adaptive white-box pilot: BPDA + EOT PGD through EnsJMG (YOLO only) -----
python phase4/p4_yolo.py     --experiment adaptive_bpda --n 300 --gpu 0

# --- checks that need no GPU -------------------------------------------------
python phase4/p4_yolo.py --preflight
python phase4/p4_florence.py --experiment clean_control --smoke-test
python tests/test_offline_logic.py
```

Shared flags: `--n/--num-images`, `--select {head,random}`, `--seed`, `--tracks`, `--gpu`,
`--device {auto,cuda,cpu}`, `--checkpoint-every`, `--no-checkpoint`, `--keep-eval-dumps`,
`--no-per-image`, plus the defense hyperparameters (defaults match Phase 3 exactly:
JPEG q75, median 3x3, TVM w=0.05/200 iters, sigma=1.0, SVD 90%, NMS IoU 0.5).

`--select head` (the default) picks the first `--n` of the sorted file list — exactly the subset the
Phase-3 attacked runs scored — so the deltas are same-subset comparable. The chosen image ids and a
SHA1 digest of them go into every summary.

### The two experiments

| Experiment | Model(s) | What it answers |
|---|---|---|
| `clean_control` | both | Do these transforms counter *attacks*, or do they just help/hurt images in general? Scores the 7 ensembles (EnsJMG, EnsJMT, Ens4W, EnsBTC, EnsBMT, EnsJBT, EnsJTS) and 6 solos (JPEG, Median, Gaussian, TVM, Blur+TVM, Bilateral — plus SVD, needed as an EnsJTS member) on clean images. **A small negative delta is an expected, publishable result**: it is the clean-accuracy cost of the transform. |
| `adaptive_bpda` | YOLO only | What happens when the attacker knows the defense? PGD eps=0.03/10 iters/alpha=eps/4 with BPDA (identity gradient for JPEG and median, true gradient for Gaussian) and EOT (mean of the three branch gradients), against EnsJMG. Reports clean / oblivious-defended / adaptive-defended mAP on the same subset. Its approximations are listed in the file header and in the summary's `implementation_uncertainty` field. |

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