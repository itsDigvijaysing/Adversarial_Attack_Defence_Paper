# Paper draft (written 2026-11-13, numbers re-checked 2026-11-15, updated 2026-11-17)

`main.tex` is a full rewrite of the rejected ECCV-W paper as an evaluation paper built on three
questions: clean cost (RQ1), dependence with confidence intervals (RQ2), and the adaptive attack
(RQ3). It uses the plain `article` class: move it into the venue template once a venue is chosen.

Build (pdfLaTeX + BibTeX; run inside `docs/paper_draft/`):
`pdflatex main && bibtex main && pdflatex main && pdflatex main`. On 17 Nov 2026: 0 errors,
0 undefined references, 0 overfull boxes, 17 pages (15 before the two figures and the new
Related Work text).

| File | What it is |
|---|---|
| `main.tex`, `refs.bib` | the paper |
| `tab_*.tex` | table bodies, pulled in by `\tabinput` (plain `\input` broke `\bottomrule`) |
| `make_tables.py` | regenerates `tab_ci_*.tex` from the committed rescore summary (stdlib only) |
| `make_figures.py`, `fig_cost_vs_patch.pdf` | Figure 2 (clean cost against the patch column), drawn from the same rescore summary; needs matplotlib. Figure 1 (the pipeline) is TikZ inside `main.tex` |
| `make_overleaf_zip.sh` | builds `overleaf_upload.zip`; `--check` compiles the zip in a temp dir |
| `main.pdf`, `main.bbl` | build outputs, kept for reading without a TeX install |

## Where every number comes from
| File | Source | Regenerate |
|---|---|---|
| `tab_ci_yolo.tex`, `tab_ci_florence.tex` (N=1000, with intervals) | `results/phase4/rescore_phase3_v2/summary_rescore_phase3_v2_latest.json` (committed) | `python docs/paper_draft/make_tables.py` |
| `tab_florence_n5000.tex` | parsed from the submitted PDF, Tab. 4; matches the committed `results/results_survey_florence_detection_*/summary_*.json` (audit of 22 Oct) | see below |
| `tab_yolo_n5000.tex`, `tab_ocr_n5000.tex` | parsed from the submitted PDF, Tabs. 3 and 5. **The summary files are not in this repo** (Digvijay's and Ankush's machines) | import the files, then rebuild from them |

The three `*_n5000.tex` files were parsed from `/home/king/Downloads/IMP Files/CV_Research_ECCV.pdf`
with a script that recomputed every recovery percentage from the mAP columns (all agree). Random
resize-and-pad is left out of the detection tables on purpose: its detection numbers are a
box-misalignment artefact. The prose numbers of Secs. 4.2, 4.3 and 5 come from the same rescore
summary as `tab_ci_*.tex` (its `conditions`, `clean_cost`, `recovery`, `best_ensemble_vs_median`
and `score_free_tables` fields).

## Correction of 19 Nov 2026
YOLO-World adaptive Median (seed 1) is 0.01648 in its summary, i.e. **0.016** at three decimals.
The abstract, Sec. 4.4 and check (v) said 0.017 (rounded twice, via 0.0165). Fixed in `main.tex`;
PDF and Overleaf zip rebuilt (22 pages, 0 overfull boxes, zip compiles standalone). Every README
results table is now generated from the summaries by `scripts/make_readme_tables.py`.
All three table generators (`make_tables.py`, `make_phase4_tables.py` and the README one) now round
half up from each value's shortest decimal form. Only `tab_smolvlm.tex` changed (accuracies are
exact multiples of 1/2000, e.g. 0.7535 -> 0.754, where the old float formatting gave 0.753); the
prose follows (Blur+TVM 0.754, Median recovery 0.093). `tab_ci_*.tex` are byte-identical.

## Number check of 15 Nov 2026
Every number in `main.tex` was traced to a file and every derived number recomputed.

| What | Source | Result |
|---|---|---|
| `tab_ci_yolo.tex`, `tab_ci_florence.tex` | `make_tables.py` re-run | byte-identical |
| `tab_yolo_n5000.tex` | ECCV PDF Tab. 3 (`pdftotext -layout`) | 20 rows, 0 mismatches |
| `tab_florence_n5000.tex` | ECCV PDF Tab. 4, and the committed `results/results_survey_florence_detection_{fgsm,pgd,patch}/summary_*.json` | 20 rows, 0 mismatches; 18 defenses x 3 attacks + clean + attacked match the JSONs at 4 dp; recovery % match the JSONs' `recovery_pct` at 1 dp |
| `tab_ocr_n5000.tex` | ECCV PDF Tab. 5 | 22 rows, 0 mismatches |
| Recovery % in the two detection tables | recomputed from the printed 4-dp mAPs | within 0.63 pp (4-dp rounding over the 0.026 / 0.034 patch denominators) |
| Phase-3 v2 N=1000 numbers | `results/results_phase3_{yolo,florence}_{fgsm,pgd,patch}_v2/summary.json` vs the rescore | 60 values, max difference 4.8e-08; YOLO PGD chain 0.4765 -> 0.0737 -> 0.4128 (EnsBTC) |
| Table 2 (attack damage) | ECCV Tab. 2; Florence also its JSONs (PGD 0.20051 -> 0.201) | all match |
| Table 1 parameters, ensemble members | `phase3_common.py` defaults, `SURVEY_ENSEMBLES` (EnsBMT is the key `ens_bilateral_median` = bilateral, median, TVM) | all match |
| Attack settings (10 PGD steps at eps/4, 35x35 centre patch, 100 Adam steps at lr 0.02, truncation + bicubic) | `archive/Phase3/run_survey_*.py`, the v2 scripts and notebooks | all match for every run in the paper. The N=1000 OCR runs (PGD at eps/10, patch at a random location) differ, and are not in the paper |
| Florence decoding (5 beams, repetition penalty 1.8, fp16), 89 synonyms, scores in [0.6, 0.98] | `phase4/p4_florence.py`, Phase-3 scripts | all match |

Prose numbers confirmed: 72% (0.7206), 0.476 -> 0.074 -> 0.364, 0.405 / 0.461 / 0.056, +0.044,
+0.049 [0.039, 0.059], +0.010 [0.002, 0.016], 0.009 / 0.005 / 0.004, 0.345 vs 0.413, 0.328,
TVM +0.331 and -0.010, clean costs 0.009 to 0.062, F1 losses 0.073 to 0.103, patch damage
0.026 / 0.034, 1.7 gray levels and 4.4x, 168 / 168 reproduced mAP values.

Corrected on 15 Nov 2026:
- TVM's clean cost on Florence-2 is 0.071 (source -0.0715), not 0.072.
- Florence-2's best defended mAP under PGD is 0.324 (EnsJMG 0.32446), not 0.325; "the same as
  under FGSM (0.328)" became "almost the same".
- "a fifth to a quarter of the clean score" is 19 to 26% (0.069 / 0.361 to 0.095 / 0.361).
- The patch removes 0.023 (YOLO) and 0.034 (Florence) mAP on the 1,000 images, not "0.02 to 0.03".
- "Ensembles beat the best single transform" was wrong: the comparison in the data is against
  Median. Under YOLO FGSM, Blur+TVM (0.345) beats every ensemble (0.336); under PGD the best
  ensemble leads Gaussian by only 0.008. Abstract, RQ2, Discussion and Conclusion now say "beat a
  median filter", and Sec. 4.3 states the best-single-transform comparison.
- "On Florence-2 ensembles buy nothing we can measure" holds for mAP only: the AP50 gaps over
  Median are 0.017 to 0.022 and their intervals exclude zero. Now stated.
- "Every transform costs YOLO-World clean accuracy" now says "each of the eight defenses whose
  clean cost we measured".
- Method text matched to `phase4/p4_yolo.py`: the Blur+TVM surrogate and the straight-through
  8-bit rounding are listed; `--pgd-select best` is described as the code does it (last vs
  best-seen iterate, compared after the 8-bit conversion; best restart across restarts); with
  `--attack-space native` the oblivious attack also runs at native resolution, so Table 5's
  oblivious columns are not Table 3's; the patch is bounded by the pixel range, not unbounded;
  the bias-corrected interval used for one claim is now defined; the `--eot-batch`
  sentence was added to the Reproducibility appendix.

## Update of 17 Nov 2026 (no new results: the A5000 runs have not started)
- **Nothing red was filled.** `results/phase4/` still holds only the August smoke tests and the
  CPU rescore, so all 25 `\pending{}` uses stand as they were.
- **Launcher sentence removed.** The appendix said the runs "are scheduled across GPUs by
  `phase4/run_queue.py`", a script that was never written. It now says every run is one command
  of `p4_yolo.py` or `p4_florence.py`. The run table below uses the plain commands
  (`docs/HANDOFF_A5000.md` has them in full).
- **Related Work, four papers read in full text (arXiv HTML) and cited:**
  - `guesmi2025drift` (arXiv 2509.24359, preprint): Tab. 2, JPEG at 0 % and BaRT at 5 to 11 %
    robust accuracy under BPDA+EOT PGD, ImageNet, eps 4/255, four classifiers. Confirmed.
  - `raber2025keep` (arXiv 2508.05489, NeurIPS 2025 workshop): Tab. 3, JPEG 70.30 % clean,
    18.70 % under straight-through BPDA, 18.72 % under white-box PGD (ResNet-50, 4/255).
  - `wang2026shared` (arXiv 2607.18339, preprint): the 13 Nov search summary was right that it
    calls the identity approximation effective on near-identity filters, but that is an argument
    from response maps (Sec. VI-B), and its Tab. XI has BPDA+EOT weakest on JPEG (54.2 % success
    against 77.6 % mean, 81.8 % Gaussian). The text says both, and names the JPEG branch as the
    part of our attack most likely to be too weak.
  - `liang2026unified` (arXiv 2610.29196, NeurIPS 2026 workshop): 11 models including YOLO-World
    and Grounding DINO, no Florence-2, no defenses. Its patches are applied digitally, so the text
    does not call them physical.
- **APSCON 2026 paper cited in the third person** (`rajput2026evaluating`). The introduction no
  longer says no measurement of input transforms exists on either model: it says the one we know
  of covers Florence-2 alone, without an adaptive attacker. **Two things to confirm by hand:**
  the author list (the conference programme prints one name for paper 6145) and the sentence
  describing its content, which was written from `docs/memory.md`, not from the PDF.
- **Two figures added.** Figure 1: the scored defense and the two attackers' gradient paths
  (TikZ). Figure 2: clean cost against patch recovery with intervals, both models; all 16 points
  lie on or above the equal-loss line (checked in the data), which is the claim of Sec. 4.3's
  first paragraph. A qualitative example is still missing (it needs the images and a model run).

## Update of 17 Nov 2026, second round
`main.pdf` and `overleaf_upload.zip` were rebuilt with these changes: 0 errors, 0 undefined
references, 0 overfull boxes, 17 pages; the zip compiles standalone.
- Sec. 3.6: a paragraph on the Florence-2 adaptive attack (pixel space, same 7/255 budget as
  YOLO-World, the least-damaged-branch objective), matching `phase4/p4_florence.py`.
- Sec. 4.4: the "YOLO-World only" scope paragraph became a red item for the Florence-2 table.
- New Sec. 4.5, "A third model": SmolVLM-256M-Instruct on yes/no object-presence questions
  scored against the COCO annotations (`phase4/p4_smolvlm.py`). The model was fixed before any
  run. Results are a red item.
- Limitations and the reproducibility appendix follow; the claim "any run can be repeated" is now
  a red item, because identical commands gave different attacked numbers on the server on 16 Nov.
- "8-bit rounding" became "8-bit conversion" (the deployed pipeline truncates).
- `make_adaptive_table.py` (new) writes `tab_adaptive_{yolo,florence}.tex` from the summaries.
  `main.tex` does not read them yet: Table 5 still has hand-placed red cells, to be replaced by
  `\tabinput{tab_adaptive_yolo}` once the file exists.
- The jobs below are superseded by `docs/HANDOFF_A5000.md` (rewritten 17 Nov): clean cost now
  runs with `--all-transforms`, and there are new jobs for Florence-2 and the third model.

## Update of 18 Nov 2026: Phase-4 results filled in
All planned Phase-4 runs finished on the RTX 5000 Ada server (17 Nov 03:36 to 18 Nov 03:10, 20
jobs, 0 failures) and are committed under `results/phase4/`. `make_phase4_tables.py` (new,
stdlib only) reads their `summary_*_latest.json` files and writes every Phase-4 table body; it
also prints each number the prose cites, with its source file. Run it from anywhere:

    python3 docs/paper_draft/make_phase4_tables.py

| Table body | Table in the PDF | Source |
|---|---|---|
| `tab_clean_n5000.tex` | clean cost of every transform, 5,000 images (Sec. 4.2) | `results/phase4/clean_control_{yolo,florence_det,florence_ocr}/` |
| `tab_adaptive_yolo.tex` | YOLO-World adaptive (Sec. 4.4); Median and EnsJMG mean (sd) over seeds 1-3, Ens4W seed 1; last column = `--eot-loss mean`, seed 1 | `results/phase4/adaptive_bpda_yolo/` |
| `tab_adaptive_florence.tex` | Florence-2 adaptive (Sec. 4.4), seed 1 | `results/phase4/adaptive_bpda_florence/` |
| `tab_smolvlm.tex` | SmolVLM-256M (Sec. 4.5), with the paired bootstrap intervals the run computed | `results/phase4/presence_qa_smolvlm/` |
| `tab_owl_survey.tex` | OWL-ViT, every transform under oblivious PGD (Sec. 4.6) | `results/phase4/oblivious_survey_owl/` (running on 18 Nov) |
| `tab_budget.tex` | defended mAP at 4/255 and 7/255, first 300 images (check iii) | `results/phase4/adaptive_bpda_{yolo,florence}_eps{4,7}of255_n300/` and `results/phase4/budget_curve_eps7of255_first300/` (7/255 scored from the 1,000-image checkpoints by `scripts/p4_subset_score.py`) |

Text changes made with the numbers:
- Sec. 3.6: K = 50, alpha = 0.002, R = 1. The claim that Eq. 3 (max over branches) is "the
  natural worst-case objective" is gone: the average was stronger on YOLO-World (EnsJMG 0.071 ->
  0.048, Ens4W 0.108 -> 0.069) and equal on Florence-2, so every worst case is the minimum over
  both objectives. Abstract and introduction say the same.
- Sec. 4.4: Table 6 and the results paragraph (worst case 0.016 / 0.048 / 0.069 against 0.018
  undefended, seed 1), checks (i) to (v), and a Florence-2 table and paragraph.
- Sec. 4.5 (SmolVLM): table and paragraph. Every ensemble is below its best member there.
- New Sec. 4.6 (OWL-ViT, fourth model): setup written; results red until the runs finish.
- Discussion, Limitations, Conclusion, reproducibility appendix (YOLO-World repeats bit for bit
  with deterministic kernels; Florence-2 differs by about 0.005 mAP between identical runs).
- Citations added and checked against arXiv: `minderer2022simple` (OWL-ViT, ECCV 2022),
  `marafioti2025smolvlm` (preprint).

## Update of 18 Nov 2026, second round: budget and fourth model
- Runs (all exited 0 by 16:49): 4/255 on the first 300 images (YOLO-World Median, JPEG,
  Gaussian, EnsJMG with both objectives; Florence-2 Median, EnsJMG) and JPEG / Gaussian alone at
  7/255; OWL-ViT-B/32 (`phase4/p4_owl.py`): oblivious survey and adaptive Median / EnsJMG on
  1,000 images, EnsJMG with the average objective on 300. 2/255 was dropped for time.
- `scripts/p4_boot_survey.py` (new) adds paired bootstrap intervals (1,000 resamples, the
  `p4_rescore.py` engine) for each ensemble against its best member, against Median, and for its
  clean cost; its output is
  `results/phase4/oblivious_survey_owl/summary_bootstrap_oblivious_survey_owl_seed1.json`.
- New tables: `tab_budget.tex` (Table 7) and `tab_owl_survey.tex` (Table 10), both from
  `make_phase4_tables.py`. Text: EnsJMG against its members (level with JPEG at 7/255, +0.011 at
  4/255), check (iii) at 4/255 on both detectors, Sec. 4.6 on OWL-ViT, the abstract's OWL-ViT
  clause, Discussion and Conclusion.
- No red item is left: `grep -n 'pending{' main.tex` finds only the header comments. Built on
  18 Nov: 0 errors, 0 undefined references or citations, 0 overfull boxes, 22 pages;
  `make_overleaf_zip.sh --check` compiles the zip standalone.

## Uploading to Overleaf
Nobody here has access to the Overleaf project (no link or login on disk), so the upload is a
manual step. Build the package first, from the repo root:

    bash docs/paper_draft/make_overleaf_zip.sh --check

It writes `docs/paper_draft/overleaf_upload.zip` with `main.tex`, `refs.bib` and the five
`tab_*.tex` files (7 files, no PDFs, no `.aux`/`.log`/`.bbl`), then unzips it into a fresh
temporary directory and compiles it with pdflatex, bibtex, pdflatex, pdflatex. On 17 Nov 2026:
8 files (the seven below plus `fig_cost_vs_patch.pdf`), 0 errors, 0 undefined references,
0 undefined citations, 0 overfull boxes, 17 pages. Add
`--with-bbl` to ship `main.bbl` too (only needed for a site that will not run BibTeX, such as
arXiv). Figures are picked up automatically once `main.tex` has `\includegraphics` lines.

**New Overleaf project (recommended):** Overleaf > New Project > Upload Project > choose
`overleaf_upload.zip`. Then Menu > Compiler: pdfLaTeX, Main document: `main.tex`, and Recompile.

**Existing project (the ECCV-W one):** this draft replaces that source, it is not a patch on it.
Keep the old project as the record of the submitted version and make a copy first (Overleaf >
Copy Project). In the copy, delete the old `.tex`, `.bib` and table files, then Upload the seven
files from the zip (`main.tex`, `refs.bib`, `tab_ci_yolo.tex`, `tab_ci_florence.tex`,
`tab_yolo_n5000.tex`, `tab_florence_n5000.tex`, `tab_ocr_n5000.tex`), set Main document to
`main.tex` and the Compiler to pdfLaTeX (XeLaTeX or LuaLaTeX are not tested, and `lmodern` plus
`microtype` are set up for pdfLaTeX). The old project's figures and style files are not used by
this draft. After any later edit on this machine, re-run the script and upload only the files
that changed (usually `main.tex` and the `tab_*.tex` files).

**Template:** the draft is still in the plain `article` class (11 pt, 1-inch margins), not the
ECCV template, because no venue is chosen yet. When one is (TMLR, or a CVPR or ICLR 2027
workshop), start from the venue's own Overleaf template and paste in the body, the tables and the
bibliography; keep `\tabinput` (plain `\input` breaks the `\bottomrule` after it) and the
`\pending` macro until every red item is filled.

## Before submitting
- The YOLO and OCR N=5000 attacked summaries no longer exist (user, 17 Nov): those two
  appendix tables stay as parsed from the submitted PDF. The YOLO clean baseline 0.4519 is
  reproduced by `results/phase4/clean_control_yolo/`.
- Confirm the APSCON 2026 entry against IEEE Xplore (authors, pages, DOI) and its one-sentence
  description in Related Work against the PDF.
- Re-check arXiv 2610.29196 for a new version (the nearest overlapping work).
- Re-check the reference entries not marked VERIFIED in `refs.bib`.
- Add one qualitative example figure (needs the images and a model run).
- Fill the remaining red items (above). `grep -n 'pending{' main.tex` should then list only
  the comment lines of the header.
