# Paper draft (written 2026-11-13)

`main.tex` is a full rewrite of the rejected ECCV-W paper as an evaluation paper built on three
questions: clean cost (RQ1), dependence with confidence intervals (RQ2), and the adaptive attack
(RQ3). It uses the plain `article` class: move it into the venue template once a venue is chosen.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`

## Where every number comes from
| File | Source | Regenerate |
|---|---|---|
| `tab_ci_yolo.tex`, `tab_ci_florence.tex` (N=1000, with intervals) | `results/phase4/rescore_phase3_v2/summary_rescore_phase3_v2_latest.json` (committed) | `python docs/paper_draft/make_tables.py` |
| `tab_florence_n5000.tex` | parsed from the submitted PDF, Tab. 4; matches the committed `results/results_survey_florence_detection_*/summary_*.json` (audit of 22 Oct) | see below |
| `tab_yolo_n5000.tex`, `tab_ocr_n5000.tex` | parsed from the submitted PDF, Tabs. 3 and 5. **The summary files are not in this repo** (Digvijay's and Ankush's machines) | import the files, then rebuild from them |

The three `*_n5000.tex` files were parsed from `/home/king/Downloads/IMP Files/CV_Research_ECCV.pdf`
with a script that recomputed every recovery percentage from the mAP columns (all agree). Random
resize-and-pad is left out of the detection tables on purpose: its detection numbers are a
box-misalignment artefact.

## What is still red in the PDF
Everything inside `\pending{...}` is a result that has not been run. Nothing red is a number.

| Placeholder | Run that fills it |
|---|---|
| Table 5, the RQ3 paragraphs, the abstract's last sentence, the conclusion's last sentence | `python phase4/p4_yolo.py --experiment adaptive_bpda --n 1000 --attack-space native --pgd-select best --pgd-iters 50 --alpha 0.002 --seed {1,2,3}` for `--target median`, `ens_jpeg_median_gaussian`, `ens_4way` |
| K, alpha, R in Sec. 3.6 | whatever the runs above use (the paper text describes `--attack-space native` and `--pgd-select best`) |
| Checks (i) to (v) in Sec. 4.4 | (i) is in every adaptive summary; (ii) `--pgd-iters 100 --n 300`; (iii) `--eps 0.1`; (iv) `--eot-loss mean`; (v) `--median-surrogate identity`. Give eps and iteration runs their own `--output-dir` |
| Clean cost on 5,000 images and for OCR (end of Sec. 4.2) | `p4_yolo.py --experiment clean_control --n 5000`; `p4_florence.py --experiment clean_control --n 1000` |

The step size 0.002 for 50 steps is a suggestion (about 2.5 * eps / steps), not something that was
tested: check convergence on a few hundred images first.

## Before submitting
- Import the YOLO and OCR N=5000 summaries and rebuild their tables from the files.
- Read the four papers listed at the end of `refs.bib` and add them to Related Work; add the
  APSCON 2026 paper in the third person.
- Re-check the reference entries not marked VERIFIED in `refs.bib`.
- Add figures (the pipeline diagram and one qualitative example): the draft has none.
