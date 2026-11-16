# Paper draft (written 2026-11-13, numbers re-checked 2026-11-15)

`main.tex` is a full rewrite of the rejected ECCV-W paper as an evaluation paper built on three
questions: clean cost (RQ1), dependence with confidence intervals (RQ2), and the adaptive attack
(RQ3). It uses the plain `article` class: move it into the venue template once a venue is chosen.

Build (pdfLaTeX + BibTeX; run inside `docs/paper_draft/`):
`pdflatex main && bibtex main && pdflatex main && pdflatex main`. On 15 Nov 2026: 0 errors,
0 warnings, 0 overfull boxes, 15 pages.

| File | What it is |
|---|---|
| `main.tex`, `refs.bib` | the paper |
| `tab_*.tex` | table bodies, pulled in by `\tabinput` (plain `\input` broke `\bottomrule`) |
| `make_tables.py` | regenerates `tab_ci_*.tex` from the committed rescore summary (stdlib only) |
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
  the bias-corrected interval used for one claim is now defined; the `--eot-batch` and
  `run_queue.py` sentence was added to the Reproducibility appendix.

## What is still red in the PDF, and the run that fills each item
Everything inside `\pending{...}` is a result that has not been run. Nothing red is a number.

All of these runs are queued by one command from the repo root, which spreads them over the
server's GPUs and adds `--gpu` and `--log-name` to each job itself:

    python phase4/run_queue.py --plan paper

The table is the reference list of what that plan contains (jobs J1 to J6). Every command runs
from the repo root and leaves `--eot-batch` at its default `auto`, which changes speed and
memory only, never the numbers. `A` stands for the adaptive flags
`--experiment adaptive_bpda --attack-space native --pgd-select best`.

| `\pending{}` in `main.tex` | Job | Command (`python phase4/p4_yolo.py A ...` unless stated) |
|---|---|---|
| Sec. 3.6: K, alpha, R; Sec. 4.4 check (ii) | J1, convergence (n = 300, EnsJMG, seed 1) | `--n 300 --target ens_jpeg_median_gaussian --seed 1 --pgd-iters 50 --alpha 0.002 --output-dir results/phase4/adaptive_bpda_yolo_iters50_n300` and the same with `--pgd-iters 100 --output-dir results/phase4/adaptive_bpda_yolo_iters100_n300`. R = 1 (the default) unless a `--restarts` run is added |
| Table 5 (caption n and all 15 cells), the RQ3 results paragraph, the abstract's last sentence, the introduction's RQ3 sentence, the conclusion's last sentence, check (i) | J2, main adaptive runs (9 jobs) | `--n 1000 --pgd-iters 50 --alpha 0.002 --target T --seed S` for T in `median`, `ens_jpeg_median_gaussian`, `ens_4way` and S in 1, 2, 3. Output: `results/phase4/adaptive_bpda_yolo/` (one `_latest` per target, seed and flag set). Check (i) is `surrogate_objective.frac_images_adaptive_below_oblivious` in each summary; the five Table 5 columns are the `headline` fields `clean_defended_mAP`, `oblivious_attacked_undefended_mAP`, `oblivious_attacked_defended_mAP`, `adaptive_attacked_undefended_mAP`, `adaptive_attacked_defended_mAP`, averaged over the three seeds |
| Check (iii), larger budget | J3 | `--n 300 --target ens_jpeg_median_gaussian --seed 1 --pgd-iters 50 --eps 0.1 --alpha 0.0067 --output-dir results/phase4/adaptive_bpda_yolo_eps0.1` (alpha scaled with eps) |
| Check (iv), max vs mean objective | J4 | J2's EnsJMG seed-1 command plus `--eot-loss mean` (own summary stem, same directory) |
| Check (v), true median gradient vs identity | J5 | J2's Median seed-1 command plus `--median-surrogate identity` (own summary stem, same directory) |
| Sec. 4.2, clean cost of every transform | J6 | `python phase4/p4_yolo.py --experiment clean_control --n 5000` and `python phase4/p4_florence.py --experiment clean_control --n 1000` (both tracks, detection and OCR) |

K = 50 and alpha = 0.002 are a suggestion that has not been tested: 50 steps of 0.002 move a
pixel at most 0.1 in total, about 3.3 eps (Madry et al.'s rule of thumb, 2.5 eps / K, would give
0.0015). J1 decides: if the 100-step run lowers defended mAP noticeably, raise K in J2 to J5
(in the launcher's plan) before running them. Eps and iteration diagnostics share the summary stem
of a normal run, which is why J1 and J3 have their own `--output-dir`. If the plan in
`phase4/run_queue.py` and this table ever disagree, fix whichever is wrong before running: the
paper text describes `--attack-space native`, `--pgd-select best`, `--eot-loss max` and the true
median gradient, so J2 must keep those.

## Uploading to Overleaf
Nobody here has access to the Overleaf project (no link or login on disk), so the upload is a
manual step. Build the package first, from the repo root:

    bash docs/paper_draft/make_overleaf_zip.sh --check

It writes `docs/paper_draft/overleaf_upload.zip` with `main.tex`, `refs.bib` and the five
`tab_*.tex` files (7 files, no PDFs, no `.aux`/`.log`/`.bbl`), then unzips it into a fresh
temporary directory and compiles it with pdflatex, bibtex, pdflatex, pdflatex. On 15 Nov 2026:
0 errors, 0 undefined references, 0 undefined citations, 0 overfull boxes, 15 pages. Add
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
- Import the YOLO and OCR N=5000 summaries and rebuild their tables from the files.
- Read the four papers listed at the end of `refs.bib` and add them to Related Work; add the
  APSCON 2026 paper in the third person.
- Re-check the reference entries not marked VERIFIED in `refs.bib`.
- Add figures (the pipeline diagram and one qualitative example): the draft has none.
- Fill every red item from J1 to J6. `grep -n 'pending{' main.tex` should then list only the
  two comment lines of the header, and the header's source list needs the adaptive summaries
  added.
