# Paper revision kit: ECCV-W EmergingAD #8 -> TMLR

**Paper:** "Benchmarking Input-Transform Defenses for Zero-Shot Detection and OCR under Oblivious
Adversarial Attacks", ECCV 2026 EmergingAD workshop #8, rejected 8 Aug 2026.
PDF: `/home/king/Downloads/IMP Files/CV_Research_ECCV.pdf` (LaTeX source on Overleaf).
**Written:** 2026-10-24. Line numbers = the PDF's margin numbers. Tables and captions have no margin
numbers, so they are cited as "Tab. N" / "Fig. N caption".

**How this file was built (not from scratch):** it starts from the 22 Oct audit in `docs/memory.md`
("Audit of the Submitted PDF", "Submission History & Reviews", "Prior Work") and re-checks every
item against the code, the committed summaries and the sources on 24 Oct. New findings from the
re-check are marked **NEW**.

**Labels:** `[verified-file]` read in a file in this repo (path:line given) or in the PDF;
`[searched]` checked online today (URL given); `[from-memory]` not re-checked today, check before
submission. No number below is new: every number is from the PDF tables, a committed
`summary*.json`, or arithmetic on those (marked "computed").

**Hard rules kept while writing:** no dataset or weights downloaded, nothing run on the GPU, no
model or COCO image loaded, no git action. Only this file was created.

---

## 0. In brief

**What the reviewers said** (exact review text is not on disk; summary from `docs/memory.md`):
- **8jGZ, 4/10, conf. 3:** reads as loosely connected observations with no research questions;
  one run per condition; no clean-image cost per defense; oblivious attacks only; OCR metric is
  consistency, not accuracy; YOLO attack loss unclear; FGSM-vs-PGD and TVM explanations not
  validated; "strictly stronger" imprecise.
- **HFW9, 6/10, conf. 2:** relevant and pragmatic, but venue fit: digital patches and classical
  filters vs real driving threats, COCO instead of driving data.

**Main problems found in the PDF itself** (Sec. 1): a wrong reference ([18] points to a finance
paper), a headline contradicted by the paper's own tables (Median is also positive in 9/9), a YOLO
loss described as something the code does not do, a "random, physically realizable" patch that is
a fixed-centre digital one, undisclosed Florence-2 pseudo-scores, two mislabelled defense
parameters (NLM, SVD), a detection result that is an evaluation artefact (Random Resize), and
seven internal number slips.

**What the revision does:** turns the paper into an evaluation paper around four research
questions (Sec. 3): RQ1 clean cost, RQ2 dependence with confidence intervals, RQ3 output-level vs
pixel-level combination at matched compute, RQ4 the adaptivity gap (BPDA+EOT). Two abstracts
(Sec. 4) cover both possible RQ4 outcomes.

---

## 1. Errata

### 1a. Factual and method errors

Long replacements are in the `R-` blocks after the table.

| ID | PDF line(s) | Current text (quoted, shortened) | Replacement | Evidence |
|---|---|---|---|---|
| E1 | 093; 410-412 | "Wang, Z., Zhang, Y., Liu, X., Chen, Z., Wang, W.: One object, multiple lies ... arXiv:2403.09761 (2024)" | Authors and ID both wrong. Use `zhao2025one` (Sec. 5); keep the `\cite` at l.093 | [searched] arXiv 2403.09761 is A. Lipton, "Hydrodynamics of Markets: Hidden Links Between Physics and Finance", q-fin.MF (https://arxiv.org/abs/2403.09761). Real paper: Zhao, Jiang, Gao, Xue, Zhao, ICCV 2025, arXiv 2507.07709 (https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_One_Object_Multiple_Lies_A_Benchmark_for_Cross-task_Adversarial_Attack_ICCV_2025_paper.html) |
| E2 | 022-025; 077-079; 305-307; 346-347 | "ensembles ... are the only defenses with positive recovery in all nine attack/model/task settings" | R-E2 | [verified-file] PDF Tabs. 3-5: Median is positive in all 9 (YOLO +28.3/+74.5/+16.1 %; F2 +52.0/+67.7/+21.5 %; OCR 0.452/0.546/0.581 vs attacked 0.346/0.294/0.383). Computed with `check_tables.py` (scratchpad): the only 9/9 defenses are Median and the 7 ensembles. **NEW:** on the committed N=1000 subset, Median is **-17.7 %** on YOLO-Patch while all 3 ensembles there stay positive (`results/results_phase3_yolo_patch_v2/summary.json`, `defenses.median.recovery_pct`) |
| E3 | 145-149 | "L the token-level cross-entropy computed against the model's own clean prediction ... applies uniformly to both output spaces" | True for Florence-2 only. YOLO uses confidence suppression. Replace with Sec. 2.2-2.3 | [verified-file] `archive/Phase3/run_survey_yolo.py:247, 270` (`loss = -cls_scores.max(dim=1)[0].sum()`), `:307` (patch: `+sum`, minimised by Adam); Florence: `run_survey_florence_detection.py:402, 440, 490` |
| E4 | 016; 045-047; 090-092; 164-167; 214 | "physically realizable Patch attacks"; "at a random location per sample ... models a physically realizable attack (e.g., a printed sticker)" | R-E4 | [verified-file] fixed centre: `phase3_common.py:717-720`, used at `run_survey_yolo.py:290`, `run_survey_florence_detection.py:472`, `run_survey_florence_ocr.py:663` (`random_location=False`). One patch per image, no EOT, no printability term (`run_survey_yolo.py:282-319`). Only the N=1000 OCR run used a random location (`results/results_patch_florence2_ocr_robust/summary.json`: `patch_random_location: true`), not the N=5000 run the paper reports |
| E5 | 186-187 | "NMS keeps the highest-confidence detection per class per region" | R-E5 + Sec. 2.8 | [verified-file] Florence `<OD>` has no confidences; every box gets `_compute_score` (geometry), `run_survey_florence_detection.py:312-322`, used at `:367-368`. Merge: `phase3_common.py:444-478` |
| E6 | 172; Tabs. 1, 3, 4, 5 rows | "SVD (90% energy)"; "SVD (90%)" | "SVD (keeps the top $\lfloor 0.9r \rfloor$ of $r$ singular values per channel, by count)" | [verified-file] `phase3_common.py:136` `k = max(1, int(S.shape[-1] * keep_ratio))`: a count, not an energy fraction |
| E7 | 172; Tabs. 1, 3, 4, 5 rows | "NLM (h=6)" | "NLM ($h = 0.8\hat\sigma$, $\hat\sigma$ the estimated noise level; 5x5 patches, search distance 6)" | [verified-file] `phase3_common.py:234-251` (`h_rel=0.8`, `h=h_rel*sigma`, `patch_size=5`, `patch_distance=6`). "h=6" survives from the Phase-2 notes (`docs/memory_phase2_archive.md:40`) |
| E8 | Tabs. 1, 3, 4 (Random Resize rows); 109; 240; 243; 343-344 | "Random Resize ... Catastrophic on detection (-0.23 mAP on YOLO-Patch)"; "random resize (-875%)" | R-E8: drop from the detection tables (or fix and rerun); keep the OCR row | [verified-file] `phase3_common.py:254-272`: shrinks by $s \sim U[0.9,1]$ and pastes at a random offset into a grey canvas of the original size; no inverse box map anywhere in the scoring path, so boxes are scored against misaligned GT |
| E9 | 019-020; 062; 072; 184-196; Tab. 1 group header; Tab. 5 group header | "7 detection-level ensembles whose per-branch detections are merged by class-aware NMS" | "7 output-level ensembles" + Sec. 2.7 (OCR uses a consensus vote) | [verified-file] `run_survey_florence_ocr.py:181-208` (`consensus_vote_text`), used at `:688` |
| E10 | 134-135 | "mapped to COCO categories through a manually constructed 70-entry lookup table" | R-E10 | [verified-file] `run_survey_florence_detection.py:259-298`: 89 keys onto 30 COCO classes (counted with `ast`); `_map_label` (`:301-309`) passes exact COCO names through first, so all 80 classes are reachable. Not a scoring defect, only the description |
| E11 | 152-154 | "[xmin, xmax] is the valid range ...: [-2.5, 2.5] in ImageNet-normalized space for Florence-2" | R-E11 | [verified-file] detection clamps to $[-2.5, 2.5]$ (`run_survey_florence_detection.py:406, 433, 444`); OCR clamps to the exact per-channel range (`run_survey_florence_ocr.py:908-909`). Computed from mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225) (local copy `/home/king/Documents/IIT Mtech/IITH_Asg/MCA/Florence-2-base/preprocessor_config.json`): valid ranges R [-2.118, 2.249], G [-2.036, 2.429], B [-1.804, 2.640]. So $[-2.5,2.5]$ is not the valid range, and it clips blue values above 0.9685 (about 247/255) by up to 0.0315, more than $\epsilon$ in pixel units. **NEW** |
| E12 | 157-158 | "eps = 0.03 normalized is ~0.007 raw, roughly 4x smaller than YOLO's budget" | "$\epsilon\sigma_c \approx 0.0068$ ($\approx 1.7/255$) vs $0.03 = 7.65/255$ for YOLO-World, $\approx 4.4\times$ smaller" | Computed: $0.03\cdot255\cdot\sigma_c$ = 1.75 / 1.71 / 1.72 levels; ratio 4.37-4.46 |
| E13 | 161-163 | "projecting back so $\lVert\hat x_{adv}-\hat x\rVert_\infty \le \epsilon$. PGD is strictly stronger than FGSM ..." | R-E13 | [verified-file] after the attack, the tensor is de-normalised, truncated to uint8 and resampled bicubically back to native size (`run_survey_florence_detection.py:408-412, 446-450`; YOLO `run_survey_yolo.py:252-253, 189-191`), so the bound holds in the attack tensor, not exactly on the evaluated image. "Strictly stronger": Sec. 6, S2 |
| E14 | 208-210 | "Defense transforms are GPU kernels (...) except JPEG which uses PIL on CPU" | R-E14 | [verified-file] `phase3_common.py:409-427`: NLM, bit-depth, random resize, dithering, anisotropic diffusion and BM3D are CPU (numpy / scikit-image / bm3d). Dithering is ordered 4x4 Bayer, 16 levels, not Floyd-Steinberg (`:282-298`); the paper does not say which |
| E15 | 210-213 | "We avoid persistent cross-run caching, which in an earlier version silently reused stale entries" | R-E15 | [verified-file] the N=5000 survey runs DO use a persistent cache, keyed by a hash of the full configuration: `phase3_common.py:540-592` (`SurveyCheckpoint`), `run_survey_yolo.py:429`, `run_survey_florence_detection.py:672-674`. The statement describes the older v2 scripts |
| E16 | 213-214 | "one run per condition, with PGD random initialization and random patch placement" | R-E15 | [verified-file] PGD start unseeded in detection (`run_survey_yolo.py:261-262`; Florence det `:432`); OCR seeds once globally (`run_survey_florence_ocr.py:852`). Patch: E4 |
| E17 | 074; 106-107; 342-344 | "Non-local means (NLM) denoising won the CAAD 2018 defense competition [22]" | R-E17 | [searched] Xie et al., CVPR 2019 (arXiv 1812.03411, https://arxiv.org/abs/1812.03411): the winner used **feature-denoising blocks inside the network** (non-local means among them) **combined with adversarial training**; "Our method was ranked first in ... CAAD 2018". Pixel-domain NLM on a frozen model does not test that result |
| E18 | 063; 113-114; 176-183 | "We tried more than ten chains ... the chains consistently over-smoothed the input and degraded detection ... We also screened GPU novel filters ... with no net gain" | R-E18 | [verified-file] no committed summary backs the screening. Only notes: `docs/memory_phase2_archive.md:108-116, 153-162` (Phase 2, FGSM, Florence-2, 1,000 images, older code; the pipelines with recorded numbers contain NLM or Random Resize, except Blur+TVM; Variant D has no recorded result). Chain and cutout code: `archive/Scripts_Extra/FGSM_Phase2_VariantY.py:517-538` |
| E19 | 103-105 | "ranking total variation minimization (TVM) [4] 'very effective' and JPEG and bit-depth reduction 'weak.'" | R-E19 | [searched] Guo et al. abstract (https://arxiv.org/abs/1711.00117): "total variance minimization and image quilting are very effective defenses in practice, in particular, when the network is trained on transformed images", gray-box and black-box settings. "weak" is not in the abstract (check the body before quoting it) |
| E20 | 199-204 | OCR on all 5,000 images, "character-level similarity ... consistency" | Add the definition and the empty-string caveat (Sec. 2.9) | [verified-file] `run_survey_florence_ocr.py:151-159`; no text-image filter anywhere in the script. Python check today: `SequenceMatcher(None,'','').ratio()` = 1.0, `('', 'abc')` = 0.0. **NEW** |
| E21 | 206-207 | "Each model runs on a single GPU. Florence-2 attacks are computed in the model's normalized pixel space" | Add: "Florence-2 runs in float16, including the attack gradients." | [verified-file] `run_survey_florence_detection.py:199` (`torch_dtype = torch.float16`). **NEW** |
| E22 | 164 | "A 35x35 RGB patch" | Say in which space: model input (Florence 768x768 resized tensor; YOLO 640 letterbox). On a 640x480 image the Florence patch covers about 29x22 native pixels after resampling | Computed: 35x640/768 = 29.2, 35x480/768 = 21.9; resize: `run_survey_florence_detection.py:501-503`. **NEW** |

#### Replacement blocks (LaTeX)

**R-E2** (abstract l.021-025, contribution l.076-079, l.305-307, l.346-347):
```latex
% l.021-025 (abstract), replaces "...that no solo filter is robust everywhere, and that
% ensembles combined at the detection-output level are the only defenses with positive
% recovery in all nine attack/model/task settings."
... that no single transform helps in every setting, and that combining transforms at the
output level is the most reliable recipe we tested: all seven output-level ensembles, and
one solo filter (the $3\times3$ median), stayed above the attacked baseline in all nine
attack/model/task settings (single-run point estimates, oblivious attacks).

% l.076-079 (contribution 3)
\item We compare combining transforms in pixel space with combining them at the output
level (one forward pass per transform; detections merged by class-aware NMS, OCR strings by
a consensus vote). All seven output-level ensembles and the median filter had positive
recovery in all nine settings; every other solo filter fell below the attacked baseline in
at least one.

% l.305-307
... the merge keeps whatever any branch recovers, which is why all seven ensembles stay
above the attacked baseline in every setting, as does the median filter alone.

% new sentence for the YOLO results (Sec. 4.5), next to l.243-245
On the first 1{,}000 images of the same split the median filter fell below the attacked
baseline on YOLO-Patch ($-17.7\%$), while the three ensembles evaluated there did not; the
sign of small Patch effects is therefore not stable across image subsets (Sec.~X gives
bootstrap intervals).
```

**R-E4** (l.164-167; also l.016, 045-047, 090-092, 214):
```latex
\paragraph{Patch~\cite{brown2017adversarial}.} A $35\times35$ RGB patch
$p\in[0,1]^{3\times35\times35}$ replaces the pixels at the centre of the model input (the
$768\times768$ processor tensor for Florence-2, the $640\times640$ letterbox for
YOLO-World) and is optimised separately for every image with 100 Adam steps (learning rate
0.02), with no per-pixel budget. The patch is digital and image-specific: there is no
expectation over placement, scale or lighting and no printability constraint, so it does
not model a physical sticker. We use it as a localised, unbounded-amplitude stress test.

% l.016:  "FGSM, PGD, and physically realizable Patch attacks"  ->
FGSM, PGD and a digital, image-specific patch attack
% l.044-047: keep the Eykholt citation as motivation only:
Patch attacks confine the change to a small region of unbounded intensity, and printed
perturbations have fooled road-sign classifiers~\cite{eykholt2018robust}; the patches we
evaluate are digital (Sec.~3.2).
% l.090-092:
Patch attacks~\cite{brown2017adversarial} optimise a small region with unbounded
intensity; physical variants exist~\cite{eykholt2018robust}, but we evaluate digital ones.
```

**R-E5** (l.186-187):
```latex
The detections of all branches are pooled, and greedy class-aware NMS keeps, within each
class, the highest-scoring box among boxes that overlap at IoU $>0.5$. For YOLO-World the
score is the model's confidence; for Florence-2 it is the geometric pseudo-score of
Sec.~3.x, so on Florence-2 the merge keeps the larger, more central box.
```

**R-E8** (Tabs. 3-4 footnote; l.109, 240, 243, 343-344):
```latex
% footnote on Tabs. 3 and 4 (remove the Random Resize rows)
Random resize-and-pad~\cite{xie2018mitigating} is omitted from the detection tables: our
implementation shrinks the image by $s\sim\mathcal{U}[0.9,1]$ and pastes it at a random
offset into a grey canvas of the original size without mapping the predicted boxes back, so
its detection score measures box misalignment, not robustness. Its OCR result is unaffected.
% l.109: "NLM and random resize are weak or harmful here"  ->
NLM is weak here (random resize is evaluated on OCR only, see Tab.~3 note)
```
Alternative if a teammate fixes the inverse map: rerun the Random Resize branch only (detection,
both models) and restore the rows.

**R-E10** (l.134-135):
```latex
Generated labels that match a COCO class name (case-insensitive) are kept as they are;
other labels are mapped through an 89-entry synonym table onto 30 COCO classes (e.g.
\emph{sofa}$\to$\emph{couch}); unmatched labels are discarded.
```

**R-E11** (l.152-154):
```latex
where $[x_{\min},x_{\max}]$ is the clipping box of the attack space: $[0,1]$ for
YOLO-World; for Florence-2 the exact per-channel image range
$[-\mu_c/\sigma_c,\,(1-\mu_c)/\sigma_c]$ for OCR and the box $[-2.5,2.5]$ for detection,
after which the image is de-normalised and clipped to $[0,1]$.
```

**R-E13** (l.161-163):
```latex
\paragraph{PGD~\cite{madry2018towards}.} Ten steps of size $\alpha=\epsilon/4=0.0075$ from
one uniform random start in the $\epsilon$-ball, projecting onto the ball and the clipping
box after every step (no restarts). The constraint holds in the attack space; the image
handed to the defense is then quantised and resampled (Sec.~3.x), so at native resolution
it holds only approximately. In every setting we ran, PGD reduced mAP or OCR consistency
more than FGSM did (Tab.~2).
```

**R-E14** (l.208-210):
```latex
TVM, median, Gaussian, SVD and bilateral filtering run as GPU kernels; JPEG (PIL), NLM
(scikit-image), bit-depth reduction, ordered dithering, anisotropic diffusion, random
resize-and-pad and BM3D run on the CPU.
```

**R-E15** (l.210-214, covers E15 and E16):
```latex
Per-image, per-condition outputs are cached in a checkpoint keyed by a hash of the complete
attack and defense configuration, so a resumed run cannot reuse outputs produced under a
different configuration (an earlier, unkeyed cache did exactly that), and every image must
carry every expected condition before it is scored. We report one run per condition; the
PGD random start is not seeded and the patch is fixed at the image centre.
```

**R-E17** (l.106-107; drop `22` from the `\cite` at l.074):
```latex
Feature denoising with non-local-means blocks inside an adversarially trained network
ranked first in the CAAD 2018 defense competition~\cite{xie2019feature}. Our NLM baseline is
a different, training-free use of the same filter on the input pixels of a frozen model, so
its weak result here does not bear on that ranking.
```

**R-E18** (l.176-183):
```latex
\paragraph{Sequential pipelines.} Chaining transforms on one image, in the spirit of
BaRT's sequential composition~\cite{raff2019barrage}, was tried in preliminary experiments
with an earlier version of our code; we do not report those runs. Section~6 compares
pixel-level and output-level combination of the same transforms directly, at matched
compute.
```
(and delete "the GPU novel filters ... with no net gain", or rerun and report them.)

**R-E19** (l.103-105):
```latex
Guo et al.~\cite{guo2018countering} studied bit-depth reduction, JPEG compression, total
variation minimization (TVM)~\cite{chambolle2004algorithm} and image quilting on ImageNet
classifiers and found TVM and quilting the most effective, in particular when the
classifier is trained on transformed images, against gray-box and black-box attacks.
```

### 1b. Internal number and claim slips (all from the paper's own tables, computed)

| ID | PDF line | Current text | Corrected text | Evidence (PDF tables) |
|---|---|---|---|---|
| N1 | 238-239 | "only Median (+16.1%) and Gaussian (+1.4%) stay positive" | "only Median (+16.1%), Gaussian (+1.4%) and SVD (+0.8%) stay above the attacked baseline, the last two by less than 0.001 mAP" | Tab. 3 Patch: SVD 0.4259 vs attacked 0.4257 |
| N2 | 256-257 | "(TVM, Gaussian, Blur+TVM) go strongly net-negative ... (-136% to -153%)" | "TVM (-136%) and Blur+TVM (-153%) go strongly net-negative under Patch, Gaussian less so (-70%)" | Tab. 4 Patch: Gaussian -70.2 |
| N3 | 257-258 | "JPEG and Median are the only solos that help on all three attacks" | "Bilateral, JPEG, Median and anisotropic diffusion help on all three attacks" | Tab. 4: Bilateral 55.4/73.5/60.6; Anisotropic 45.2/65.7/20.4 |
| N4 | 262-263 | "The absolute spread among ensembles is small (<0.01 mAP on FGSM/PGD)" | "Six of the seven ensembles lie within 0.005 mAP of each other on FGSM and PGD; EnsJTS trails by up to 0.022" | Tab. 4: spread with EnsJTS 0.0150 (FGSM), 0.0216 (PGD); without 0.0045, 0.0034 |
| N5 | 283-285 | "Ens4W ... OD within a point of EnsJMG" | "mean OD recovery 3.1 points below EnsJMG (49.5 vs 52.6%) and a lower worst case (17.0 vs 29.0%)" | Tab. 6 |
| N6 | 285-287 | "EnsJMT ... the best three-branch option on OCR-FGSM/PGD" | "the best three-branch option on OCR-FGSM (0.555); on OCR-PGD EnsJBT (0.661) and EnsBMT (0.651) are slightly higher" | Tab. 5: EnsJMT PGD 0.649 |
| N7 | 279-280; Tab. 1 EnsJMG row | "leads both Patch-OD settings" | "leads YOLO-Patch; on Florence-Patch EnsJBT, EnsBMT and the Bilateral filter alone score higher" | Tab. 4 Patch: Bilateral 0.3165, EnsJBT 0.3105, EnsBMT 0.3099, EnsJMG 0.3057 |
| N8 | Tab. 1 Random Resize row | "Recovers only OCR" | Row removed (E8); if kept: "also +17.7% on YOLO-PGD" | Tab. 3 PGD 0.1311 vs 0.0619 |
| N9 | Tab. 1 Anisotropic row | "weak or negative on YOLO and OCR-Patch" | "weak on YOLO-FGSM (+4.1%), negative on YOLO-Patch and OCR-Patch, +61.2% on YOLO-PGD" | Tab. 3; Tab. 5 Patch 0.372 < 0.383 |
| N10 | 302-303 | "TVM wins OCR" | "TVM wins OCR-FGSM and OCR-PGD; Median wins OCR-Patch" | Tab. 5 |
| N11 | Tab. 3, Random Resize | 0.1311 under FGSM **and** PGD (identical to 4 decimals) | Teammate check (Sec. 7) | Possible copy error; Florence has 0.1389 vs 0.1329 for the same pair |

Re-verified and **correct** as printed: Tab. 4 (all cells match the committed
`results/results_survey_florence_detection_{fgsm,pgd,patch}/summary_*.json`); Tab. 6 (recomputed
from Tabs. 3-5, matches to rounding: e.g. EnsJMG 52.6 / 29.0 / 34.9 / 46.7); l.222-227 attack
drops (0.390, 0.130, 0.247, 0.081, 0.026, 0.034); l.242-243 (+0.011, -0.229 mAP); "EnsJMT edges
F2 PGD" (0.2961, top); "PGD recovery exceeds FGSM recovery" holds for every row of Tabs. 3-5.

### 1c. Flags for teammates (cannot be checked here)

- **YOLO N=5000 numbers (Tab. 3) and OCR N=5000 numbers (Tab. 5)** have no committed file
  (`docs/memory.md`, "Headline Results"). They are real runs on other machines; the summaries need
  importing before the revised paper cites them.
- **0.1311** (N11): open the YOLO `summary_fgsm.json` and `summary_pgd.json`,
  `defenses.random_resize.mAP`.
- Confirm the YOLO N=5000 run used `archive/Phase3/run_survey_yolo.py` as committed (the
  equations in Sec. 2 describe that file).

---

## 2. Method rewrites (LaTeX)

Drop-in replacement for Sec. 3.2 (l.142-167) plus the metric and implementation paragraphs.
Cite keys are suggestions: map them to the Overleaf `.bib`.

### 2.1 Threat model and pipeline

```latex
\subsection{Threat model and attack pipeline}
All attacks are crafted against the undefended model; the defense is applied afterwards
(oblivious setting~\cite{athalye2018obfuscated,carlini2019evaluating}). Section~7 adds an
adaptive attacker. Each attack runs in the model's own input space and is then delivered as
an ordinary 8-bit image, which is what a deployed defense receives:
\begin{itemize}
\item \textbf{Florence-2.} The processor $P$ resizes the image to $768\times768$ (bicubic,
aspect ratio not preserved) and normalises each channel with the ImageNet mean $\mu$ and
standard deviation $\sigma$. The attack perturbs $z=P(x)$; the result is de-normalised,
clipped to $[0,1]$, truncated to 8 bits and resized back to the native $H\times W$
(bicubic). The model runs in float16, including the attack gradients.
\item \textbf{YOLO-World.} The image is letterboxed to $640\times640$ (bilinear resize, grey
padding 114); the attack perturbs this raw $[0,1]$ tensor $z=\ell(x)$; the result is
truncated to 8 bits, the padding is cropped, and the crop is resized back to $H\times W$
(bicubic).
\end{itemize}
Writing $Q(v)=\lfloor 255v\rfloor/255$ for the 8-bit truncation and $R$ for the resize, the
image passed to the defense is $x_{\mathrm{adv}} = R\big(Q(D(z_{\mathrm{adv}}))\big)$, where $D$
de-normalises (Florence-2) or crops the letterbox (YOLO-World). The $\ell_\infty$ constraint
therefore holds for $z_{\mathrm{adv}}$ and only approximately for $x_{\mathrm{adv}}$.
```

### 2.2 Florence-2 objective (self-labelled token cross-entropy)

```latex
\paragraph{Florence-2 objective.} For a task prompt $q\in\{\texttt{<OD>},\texttt{<OCR>}\}$
we first decode the clean input with the deployed decoding (beam search, 5 beams,
repetition penalty 1.8, length penalty 1.0) to obtain the model's own output tokens
$y^{\mathrm{s}}=(y^{\mathrm{s}}_1,\dots,y^{\mathrm{s}}_T)$. The attack maximises the mean
token-level cross-entropy of these self-labels under teacher forcing:
\begin{equation}
\mathcal{L}_{\mathrm{F2}}(z') = -\frac{1}{T}\sum_{t=1}^{T}
  \log p_\theta\big(y^{\mathrm{s}}_t \,\big|\, y^{\mathrm{s}}_{<t},\, z',\, q\big).
\end{equation}
No ground truth is used: a deployed zero-shot service exposes none to an attacker.
```
Source: `run_survey_florence_detection.py:384-402` (FGSM), `:416-444` (PGD), `:454-490` (patch:
`(-out.loss).backward()` with Adam, i.e. CE is maximised); OCR the same at
`run_survey_florence_ocr.py:250-428` [verified-file].

### 2.3 YOLO-World objective (confidence suppression)

```latex
\paragraph{YOLO-World objective.} YOLO-World is prompted once with the 80 COCO class names.
Let $s_{a,c}(z)\in[0,1]$ be the class score (after the sigmoid) of anchor
$a\in\{1,\dots,8400\}$ for class $c$, read from the detection head before non-maximum
suppression. The attack maximises
\begin{equation}
\mathcal{L}_{\mathrm{YW}}(z') = -\sum_{a=1}^{8400} \max_{c}\; s_{a,c}(z'),
\end{equation}
i.e.\ it suppresses the most confident class at every anchor. It uses no labels (neither
ground truth nor self-labels), no box-regression term and no objectness term (YOLOv8 heads
are anchor-free and have none).
```
Source: `run_survey_yolo.py:243-247, 266-270, 303-307`; prompt `:635` (`model.set_classes`);
`pred[:, 4:, :]` = class rows of the `[1, 84, 8400]` head output [verified-file]. "After the
sigmoid" relies on ultralytics' Detect head applying `sigmoid` in inference mode [from-memory].

### 2.4 FGSM, PGD and Patch updates

```latex
With $\mathcal{L}\in\{\mathcal{L}_{\mathrm{F2}},\mathcal{L}_{\mathrm{YW}}\}$, clipping box
$\mathcal{C}$ (Sec.~3.x) and $\epsilon=0.03$ in the attack space:
\begin{align}
z^{\mathrm{FGSM}} &= \Pi_{\mathcal{C}}\big(z+\epsilon\,\mathrm{sign}\,\nabla_z\mathcal{L}(z)\big),\\
z^{k+1} &= \Pi_{\mathcal{C}\cap B_\infty(z,\epsilon)}
   \big(z^{k}+\alpha\,\mathrm{sign}\,\nabla\mathcal{L}(z^{k})\big),\qquad
   z^{0}=\Pi_{\mathcal{C}}(z+u),\; u\sim\mathcal{U}[-\epsilon,\epsilon]^{d},
\end{align}
with $\alpha=\epsilon/4$ and $K=10$ steps (one random start, no restarts). The patch attack
optimises a $35\times35$ patch pasted at the centre of $z$ with 100 Adam steps (learning
rate 0.02), initialised uniformly in $[0.25,0.75]$ (detection) or $[0,1]$ (OCR),
maximising $\mathcal{L}_{\mathrm{F2}}$ or, for YOLO-World, minimising
$\sum_a\max_c s_{a,c}$.
```

### 2.5 Budgets per model

| | Florence-2 | YOLO-World |
|---|---|---|
| Attack space | normalised, 768x768 resized tensor | raw [0,1], 640x640 letterbox |
| $\epsilon$ in attack space | 0.03 | 0.03 |
| $\epsilon$ in 8-bit levels | $0.03\cdot255\cdot\sigma_c$ = 1.75 / 1.71 / 1.72 (R/G/B) | 7.65 |
| Ratio | 1 | about 4.4x larger |
| PGD $\alpha$, steps | 0.0075, 10 | 0.0075, 10 |
| Clip box | det: [-2.5, 2.5]; OCR: exact per-channel range | [0, 1] |
| Patch | 35x35 in the 768x768 tensor (about 0.21% of it) | 35x35 in the letterbox (about 0.30% of it) |

```latex
Because Florence-2 is attacked in normalised space, the same $\epsilon=0.03$ is
$\epsilon\sigma_c\approx1.7/255$ per pixel, about $4.4\times$ smaller than YOLO-World's
$7.65/255$. Attack strength is therefore not comparable across models, and all conclusions
are within-model.
```
(Patch shares computed: 1225/589824 = 0.21%; 1225/409600 = 0.30%.)

### 2.6 Defense parameters (replaces l.169-175; exact code values)

```latex
\begin{table}[t]\centering\small
\caption{Input transforms and their parameters (identical for both models, not tuned).}
\begin{tabular}{@{}ll@{}}\toprule
Transform & Parameters \\ \midrule
JPEG & quality 75 (PIL) \\
Median & $3\times3$, reflect padding \\
Gaussian & $\sigma=1$, separable, 7 taps, reflect padding \\
Bit-depth~\cite{xu2018feature} & 4 bits per channel, floor quantisation \\
Ordered dithering & $4\times4$ Bayer matrix, 16 levels per channel \\
TVM~\cite{chambolle2004algorithm} & weight 0.05, 200 dual-projection iterations, $\tau=0.25$ \\
Blur+TVM & Gaussian ($\sigma=1$) then TVM \\
NLM & scikit-image fast mode, $h=0.8\hat\sigma$, $5\times5$ patches, search distance 6 \\
SVD & per channel, keep the top $\lfloor0.9r\rfloor$ of $r$ singular values \\
Bilateral & $\sigma_s=1$, $\sigma_r=0.1$ (on $[0,1]$), $5\times5$ window \\
Anisotropic diffusion & Perona--Malik, 5 iterations, $\kappa=0.05$, $\gamma=0.2$ \\
BM3D (OCR only) & $\sigma_{\mathrm{psd}}=0.1$ \\
Random resize-and-pad & scale $\mathcal{U}[0.9,1]$, bicubic, grey canvas, seeded per image \\
\bottomrule\end{tabular}\end{table}
```
Source: `phase3_common.py:29-35, 95-345` [verified-file]. Note: the paper calls bit-depth and
median "classical squeezing" without citing Xu et al. (NDSS 2018); add `xu2018feature`.

### 2.7 Combining branches: NMS for detection, consensus vote for OCR

```latex
\paragraph{Output-level ensembles.} An ensemble runs the model once per branch.
\emph{Detection:} the branches' detections are pooled and greedy class-aware NMS keeps,
within each class, the highest-scoring box among boxes overlapping at IoU $>0.5$.
\emph{OCR:} strings cannot be merged by NMS, so we use a consensus vote: if one normalised
string is produced by at least two branches (and no other string ties it), it is returned;
otherwise we return the branch output with the highest summed similarity $s$
(Eq.~\ref{eq:ocr}) to the other branches, breaking ties by length.
```
Source: `phase3_common.py:444-478`; `run_survey_florence_ocr.py:181-208` [verified-file].

### 2.8 Florence-2 pseudo-score disclosure

```latex
\paragraph{Florence-2 detection scores.} Florence-2's \texttt{<OD>} output contains boxes
and labels but no confidences, whereas COCO mAP needs a ranking. We give each box $b$ a
deterministic geometric pseudo-score
\begin{equation}
\hat s(b)=\mathrm{clip}\Big(0.6+0.2\min\big(\tfrac{A_b}{A_I},0.5\big)+0.15\,(1-d_b),\;0.6,\;0.98\Big),
\end{equation}
where $A_b/A_I$ is the box-to-image area ratio and $d_b$ the distance from the box centre to
the image centre, normalised by image width and height. Every condition is scored with the
same rule, so mAP differences between conditions are comparable; absolute Florence-2 mAP,
however, ranks boxes by size and centrality rather than by model confidence and must not be
compared with YOLO-World's. A defense that systematically enlarges or shrinks boxes also
changes their ranking, so we additionally report score-free precision, recall and F1 at
IoU $0.5$. The same scores drive Florence-2's per-image NMS and the ensemble merge.
```
Source: `run_survey_florence_detection.py:312-322` (formula), `:335-347, 367-368` (NMS on these
scores) [verified-file]. Score-free metrics: `phase4/p4_rescore.py:447` (`score_free_counts`).

### 2.9 OCR metric: self-consistency

```latex
\paragraph{OCR self-consistency.} COCO has no text ground truth, so we measure how well a
defense preserves the model's own clean reading. For an image with clean OCR output $o$ and
output $o'$ under some condition,
\begin{equation}\label{eq:ocr}
s(o,o')=\frac{2M(\bar o,\bar o')}{|\bar o|+|\bar o'|},\qquad s=1 \text{ if both are empty},
\end{equation}
where $\bar\cdot$ lower-cases, removes tags and collapses whitespace, and $M$ counts the
characters matched by Python's \texttt{difflib.SequenceMatcher}. We report the mean of $s$
over images (\emph{self-consistency}; the clean value is 1 by construction) and recovery as
for mAP. This is not OCR accuracy: a wrong clean reading that survives the attack scores 1.
We do not filter for images that contain text, so an image whose clean output is empty
scores 1 under any condition that also outputs nothing and 0 under any condition that
outputs text; we report the share of such images (App.~X).
```
Source: `run_survey_florence_ocr.py:151-159, 637-694`; the 1.0 / 0.0 behaviour checked in
Python today [verified-file].

---

## 3. New structure

### 3.1 Introduction opening (LaTeX)

```latex
\section{Introduction}
Open-vocabulary detectors such as YOLO-World~\cite{cheng2024yolo} and unified
vision-language models such as Florence-2~\cite{xiao2024florence} are increasingly used as
frozen components: a deployer queries them with text and cannot retrain them. If such a
model faces adversarial inputs, the defenses available without weights, training data or
fine-tuning are transformations of the input image. On image classifiers these defenses
have a mixed record: they look effective against attacks that ignore
them~\cite{guo2018countering,xu2018feature} and largely fail once the attacker
differentiates through them~\cite{athalye2018obfuscated,tramer2020adaptive,croce2022evaluating}.
On closed-set detectors, JPEG and smoothing baselines lose much of their benefit under
BPDA~\cite{liu2022segment}, and many dedicated patch defenses are bypassed by adaptive
attacks~\cite{zheng2025revisiting}.

Whether this record carries over to open-vocabulary models has not been measured: their
robustness has been studied under distribution shift~\cite{chhipa2024open} and new attacks
on unified models~\cite{zhao2025one}, not under defenses. We close this gap with an
evaluation organised around four questions:
\begin{description}
\item[RQ1 (cost)] How much clean accuracy does each transform cost, per model and task?
\item[RQ2 (dependence)] How much does a transform's benefit depend on the model, the task
  and the attack, once image-level uncertainty is accounted for?
\item[RQ3 (combination)] Does combining transforms at the output level (one forward pass
  per transform, merged outputs) beat combining them in pixel space at matched compute?
\item[RQ4 (adaptivity)] How much of the benefit measured against oblivious attacks survives
  an attacker who differentiates through the defense (BPDA with EOT over the branches), when
  each system is scored at its own worst case?
\end{description}
\paragraph{Contributions.} (i) An evaluation protocol for training-free input defenses on
open-vocabulary detection and OCR that reports clean cost, paired-bootstrap intervals and
worst-case adaptive robustness side by side. (ii) Results for 13 transforms and 7
output-level ensembles on YOLO-World and Florence-2 under FGSM, PGD and a digital patch on
COCO val2017. (iii) A matched-compute comparison of output-level and pixel-level
combination. (iv) An adaptive BPDA+EOT evaluation showing that [RQ4 result: Sec.~4,
abstract A or B]. Code and per-image outputs will be released.
```
Gap claim status: no paper found that evaluates input-transform defenses on YOLO-World or
Florence-2, oblivious or adaptive [searched 2026-10-24: arXiv API `all:"YOLO-World" AND
all:adversarial` returned 2 hits, neither a defense; `all:"Florence-2" AND all:adversarial`
returned 2507.07709 (the ICCV 2025 paper) and 2608.02137, both attack papers; one web search,
nothing matching]. Re-run the search the week before submission.

### 3.2 RQ -> evidence map

| RQ | Citable now (PDF / committed) | CPU only, no GPU, no images | GPU runs (in order) | New code needed | Paper output |
|---|---|---|---|---|---|
| RQ1 clean cost | Nothing in the PDF. The N=1000 v2 dumps hold `clean_<def>.json` for 5 solos + 3 ensembles, both detectors [verified-file `results/results_phase3_*_v2/`] | `python phase4/p4_rescore.py` (reads dumps + `annotations/instances_val2017.json`, which is on disk; loads no model or image). **NEW:** the N=5000 survey scripts scored `clean+<def>` for every defense (`run_survey_yolo.py:459-465`) but the summaries dropped it (`:553-577`); the teammates' `checkpoint_{attack}.pkl` / `clean_*.json` would give N=5000 clean cost for all 19 defenses on CPU | `p4_yolo.py --experiment clean_control --n 5000`; `p4_florence.py --experiment clean_control --tracks florence_det florence_ocr --n 1000` | None | Clean-cost table per model/task; "recovery minus cost" column |
| RQ2 dependence + CIs | Tabs. 3-5 point estimates; N=1000 vs N=5000 sign flip of Median on YOLO-Patch (E2) | `p4_rescore.py`: paired image-level bootstrap (B=1000) of defended-attacked, clean cost, best ensemble minus Median; score-free P/R/F1 for Florence | 3 PGD seeds for YOLO (seeded init exists in `p4_yolo.py`); OCR needs per-image similarities (the N=5000 OCR state keeps sums only, `run_survey_florence_ocr.py:690-694`) | Port the Florence-2 attacks into `phase4/` with per-image logging (they live only in `archive/Phase3/`) | CI table / forest plot per setting |
| RQ3 combination at matched compute | Tab. 6 (ensembles only). The screening claim (l.176-183) has no committed numbers (E18) | none | Per model, PGD + Patch, n=1000: (a) EnsJMG, 3 passes; (b) pipeline JPEG->Median->Gaussian, 1 pass; (c) pixel average of the 3 branch images, 1 pass; (d) best solo, 1 pass; (e) control: 3 non-denoising views (identity, h-flip, 0.9 rescale; boxes mapped back) merged by the same NMS, 3 passes. (e) separates "merging views" from "denoising" | Yes: (b), (c), (e) are not in `phase4/` | Recovery vs forward passes and ms/image |
| RQ4 adaptivity gap | None (Sec. 5 defers it). Smoke test n=2 is not citable | none | `python phase4/p4_yolo.py --experiment adaptive_bpda --target ens_jpeg_median_gaussian --eot-loss max --n 1000 --seed {1,2,3}`; same with `--target median` and `--target ens_4way`; convergence check `--pgd-iters 50 --n 300`; a large-eps sanity run (mAP must collapse); ablations `--quantize round`, `--attack-space native`, `--restarts 3` | Florence-2 adaptive attack (none yet). Without it, scope the title and RQ4 to YOLO-World | Oblivious vs adaptive table, each system at its worst case: defended = min over attacks, undefended = min over attacks |

Notes for RQ4 (current code state, uncommitted on 24 Oct): BPDA backward = identity for JPEG and
TVM, true a.e. gradient for median by default (`--median-surrogate identity` gives the old
identity), true gradient for Gaussian; `--eot-loss max` (default) suppresses the most confident
branch per anchor, matching the NMS union; `--eot-loss mean` is the earlier pilot
(`phase4/p4_yolo.py:329-339, 1054-1083`) [verified-file]. Cost estimate for the core GPU runs:
11-16 h on the RTX 3060 or 5-7 h on one A6000 [from-memory, `docs/memory.md` Phase-4 Status].

### 3.3 Proposed outline (TMLR)

1. Introduction (problem, gap, RQ1-RQ4, contributions). Do not call it a survey: TMLR stopped
   taking surveys on 1 Sep 2026 [from-memory, `docs/memory.md`].
2. Related work (adaptive evaluation; input transforms; detector patch defenses; open-vocab
   robustness).
3. Setup: models, data, threat models, pipeline (Sec. 2.1), objectives (2.2-2.4), budgets (2.5),
   defenses (2.6), combination (2.7), metrics (2.8-2.9), protocol (seeds, bootstrap).
4. RQ1 clean cost. 5. RQ2 oblivious robustness with CIs. 6. RQ3 combination. 7. RQ4 adaptive.
8. Limitations: digital patch only; COCO, not driving data; OCR self-consistency; pseudo-scores;
   one detector per family; within-model conclusions only.
9. Conclusion. Appendix: full Tabs. 3-5, parameters, empty-OCR share, per-seed results.

### 3.4 Reviewer point -> where it is fixed

| Reviewer point | Fix |
|---|---|
| No research questions, loose observations (8jGZ) | Sec. 3.1-3.3 |
| One run per condition (8jGZ) | RQ2 bootstrap + 3 PGD seeds |
| No clean cost per defense (8jGZ) | RQ1 |
| Oblivious only (8jGZ) | RQ4 |
| OCR metric is consistency (8jGZ) | Named "self-consistency", defined, empty-string caveat (2.9). Optional: a text-bearing subset with ground truth (e.g. COCO-Text over the overlapping images [from-memory, check overlap]); needs a new download, so only with the user's go-ahead |
| YOLO loss unclear (8jGZ) | 2.3 |
| FGSM-vs-PGD and TVM explanations unvalidated (8jGZ) | Sec. 6, S1 and S10 |
| "Strictly stronger" (8jGZ) | R-E13, S2 |
| Venue fit: digital patch, classical filters, COCO (HFW9) | TMLR / WACV E&D; patch called digital; AV framing reduced to motivation; driving data in limitations |

---

## 4. Abstract drafts (placeholders only for new results)

Placeholders: **[X]** = the RQ4 headline gap, **[Y]** = the clean cost, plus named ones where a
number is not known yet. Numbers that do appear come from the PDF tables (N=5000, oblivious).

**(A) Ensembles keep measurable worst-case robustness**
```latex
\begin{abstract}
Open-vocabulary detectors are increasingly deployed as frozen models, which leaves
transformations of the input image as the main training-free defense. We evaluate 13 input
transforms and 7 output-level ensembles on two structurally different open-vocabulary
models, YOLO-World and Florence-2 (detection and OCR), on COCO val2017 under FGSM, PGD and a
digital patch attack. For every defense we report the clean-image cost, paired-bootstrap
confidence intervals, and robustness against both oblivious attacks and an adaptive
attacker that differentiates through every branch with BPDA and EOT. Single transforms help
in some settings and hurt in others. Ensembles that run the model once per transform and
merge the outputs are the most reliable: against the adaptive attacker the best ensemble
keeps [X] mAP above the undefended model's worst case on YOLO-World, at a clean cost of [Y]
mAP and [K] times the inference compute, and combining at the output level beats combining
the same transforms in pixel space at matched compute [RQ3 result]. We release code and
per-image outputs.
\end{abstract}
```

**(B) Oblivious evaluation overstates robustness**
```latex
\begin{abstract}
Input-transform defenses have repeatedly looked strong against attacks that ignore them
and then failed against attacks that do not. We ask whether open-vocabulary detectors
repeat this pattern. We evaluate 13 input transforms and 7 output-level ensembles on
YOLO-World and Florence-2 (detection and OCR) on COCO val2017 under FGSM, PGD and a
digital patch attack, reporting clean cost and paired-bootstrap intervals. Against
oblivious PGD, the best ensembles recover most of the lost mAP on YOLO-World. An adaptive
attacker that differentiates through all branches with BPDA and EOT removes most of that
gain: scored at its worst case, the defended model is only [X] mAP above the undefended one,
so oblivious evaluation overstates input-transform robustness by [GAP] mAP, while the
defenses cost [Y] mAP on clean images. We recommend that robustness claims for
open-vocabulary detectors report adaptive, worst-case numbers and clean cost together.
\end{abstract}
```
If (B) holds, the title can say so, e.g. "Oblivious Evaluation Overstates Input-Transform
Robustness of Open-Vocabulary Detectors". If (A) holds: "How Much Do Input Transforms Protect
Open-Vocabulary Detectors? Clean Cost, Uncertainty and Adaptive Attacks".

---

## 5. References

### 5.1 Fixes to existing entries

| Ref | Problem | Fix | Status |
|---|---|---|---|
| [18] | Wrong authors and wrong arXiv ID (finance paper) | `zhao2025one` below | [searched] arXiv + CVF |
| [22] | Entry itself is correct (Xie, Wu, van der Maaten, Yuille, He, CVPR 2019); the text citing it (l.106) misstates what won CAAD 2018 | Keep entry; fix text with R-E17; remove from l.074 | [searched] https://arxiv.org/abs/1812.03411 |
| [8] | Venue correct (ICLR 2018); the text's "weak" quote is unverified | R-E19 | [searched] https://iclr.cc/virtual/2018/poster/139 |
| [7] | Cited as arXiv 2014 | Published at ICLR 2015 | [from-memory] |
| [1], [10], [17] | Correct | none | [searched] arXiv abs pages: ICML 2018; CVPR 2022 ("camera ready"); NeurIPS 2020 ("thirteen defenses") |
| others | Not re-checked today | none | [from-memory] |

### 5.2 Entries to add (each verified today)

| Key | Title, authors | Venue | Verified via |
|---|---|---|---|
| zhao2025one | One Object, Multiple Lies: A Benchmark for Cross-task Adversarial Attack on Unified Vision-Language Models; Zhao, Jiang, Gao, Xue, Zhao | ICCV 2025, pp. 187-196 | arXiv abs + CVF open access (pages from CVF/mlanthology via search: confirm) |
| croce2022evaluating | Evaluating the Adversarial Robustness of Adaptive Test-time Defenses; Croce, Gowal, Brunner, Shelhamer, Hein, Cemgil | ICML 2022, PMLR 162:4421-4435 | arXiv abs ("ICML'22") + https://proceedings.mlr.press/v162/croce22a.html |
| xu2018feature | Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks; Xu, Evans, Qi | NDSS 2018 | arXiv abs ("To appear in NDSS 2018"); DOI from a search hit: confirm |
| kurakin2017adversarial | Adversarial Machine Learning at Scale; Kurakin, Goodfellow, Bengio | ICLR 2017 | arXiv abs; camera-ready header "Published as a conference paper at ICLR 2017" (OpenReview page blocked by a bot check) |
| athalye2018obfuscated | Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples; Athalye, Carlini, Wagner | ICML 2018 | arXiv abs ("ICML 2018"). Already ref [1]; keep |
| chhipa2024open | Open-Vocabulary Object Detectors: Robustness Challenges under Distribution Shifts; Chhipa, De, Chippa, Saini, Liwicki | ECCV 2024 Workshops, LNCS, pp. 62-79 | arXiv abs ("Accepted at ECCVW 2024"); pages/DOI from DBLP via search: confirm |
| zheng2025revisiting | Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights; Zheng, Sun, Lin, Zhao, Ma, Zhang, Wang, Wang, Shen | ICCV 2025, pp. 23476-23486 | arXiv abs ("Accepted by ICCV 2025") + CVF open access (pages via search: confirm) |
| jing2024pad | PAD: Patch-Agnostic Defense against Adversarial Patch Attacks; Jing, Wang, Ren, Dong, Zou | CVPR 2024, pp. 24472-24481 | arXiv abs ("Accepted by CVPR 2024") + CVF open access |
| croce2020reliable | Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks; Croce, Hein | ICML 2020, PMLR 119:2206-2216 | arXiv abs ("In ICML 2020") + https://proceedings.mlr.press/v119/croce20b.html |
| apscon2026 | Self-citation placeholder | IEEE APSCON 2026 | User to fill from the camera-ready |

URLs: https://arxiv.org/abs/2507.07709 ; https://arxiv.org/abs/2202.13711 ;
https://arxiv.org/abs/1704.01155 ; https://arxiv.org/abs/1611.01236 ;
https://arxiv.org/abs/1802.00420 ; https://arxiv.org/abs/2405.14874 ;
https://arxiv.org/abs/2508.00649 ; https://arxiv.org/abs/2404.16452 ;
https://arxiv.org/abs/2003.01690 ;
https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Revisiting_Adversarial_Patch_Defenses_on_Object_Detectors_Unified_Evaluation_Large-Scale_ICCV_2025_paper.html ;
https://openaccess.thecvf.com/content/CVPR2024/html/Jing_PAD_Patch-Agnostic_Defense_against_Adversarial_Patch_Attacks_CVPR_2024_paper.html ;
https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2017/adversarial_machine_learning_at_scale__b7aefcef.txt

```bibtex
@inproceedings{zhao2025one,
  title     = {One Object, Multiple Lies: A Benchmark for Cross-task Adversarial Attack on Unified Vision-Language Models},
  author    = {Zhao, Jiale and Jiang, Xinyang and Gao, Junyao and Xue, Yuhao and Zhao, Cairong},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {187--196},
  year      = {2025}
}
@inproceedings{croce2022evaluating,
  title     = {Evaluating the Adversarial Robustness of Adaptive Test-time Defenses},
  author    = {Croce, Francesco and Gowal, Sven and Brunner, Thomas and Shelhamer, Evan and Hein, Matthias and Cemgil, Taylan},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {162},
  pages     = {4421--4435},
  year      = {2022}
}
@inproceedings{xu2018feature,
  title     = {Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks},
  author    = {Xu, Weilin and Evans, David and Qi, Yanjun},
  booktitle = {Network and Distributed System Security Symposium (NDSS)},
  year      = {2018},
  doi       = {10.14722/ndss.2018.23198}
}
@inproceedings{kurakin2017adversarial,
  title     = {Adversarial Machine Learning at Scale},
  author    = {Kurakin, Alexey and Goodfellow, Ian and Bengio, Samy},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017}
}
@inproceedings{athalye2018obfuscated,
  title     = {Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples},
  author    = {Athalye, Anish and Carlini, Nicholas and Wagner, David},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning (ICML)},
  year      = {2018}
}
@inproceedings{chhipa2024open,
  title     = {Open-Vocabulary Object Detectors: Robustness Challenges under Distribution Shifts},
  author    = {Chhipa, Prakash Chandra and De, Kanjar and Chippa, Meenakshi Subhash and Saini, Rajkumar and Liwicki, Marcus},
  booktitle = {Computer Vision -- ECCV 2024 Workshops},
  series    = {Lecture Notes in Computer Science},
  pages     = {62--79},
  year      = {2024},
  doi       = {10.1007/978-3-031-91672-4_5}
}
@inproceedings{zheng2025revisiting,
  title     = {Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights},
  author    = {Zheng, Junhao and Sun, Jiahao and Lin, Chenhao and Zhao, Zhengyu and Ma, Chen and Zhang, Chong and Wang, Cong and Wang, Qian and Shen, Chao},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {23476--23486},
  year      = {2025}
}
@inproceedings{jing2024pad,
  title     = {{PAD}: Patch-Agnostic Defense against Adversarial Patch Attacks},
  author    = {Jing, Lihua and Wang, Rui and Ren, Wenqi and Dong, Xin and Zou, Cong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {24472--24481},
  year      = {2024}
}
@inproceedings{croce2020reliable,
  title     = {Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks},
  author    = {Croce, Francesco and Hein, Matthias},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {2206--2216},
  year      = {2020}
}
% Self-citation: cite in the third person ("[AUTHORS] evaluated ..."), never "our previous work".
@inproceedings{apscon2026,
  title     = {Evaluating and Mitigating Adversarial Vulnerabilities in Vision Foundation Models},
  author    = {[AUTHORS AS PRINTED]},
  booktitle = {Proceedings of the IEEE Applied Sensing Conference (APSCON)},
  year      = {2026},
  pages     = {[PAGES]},
  doi       = {[DOI]}
}
```
APSCON title from `docs/memory.md` [verified-file]; the expansion "Applied Sensing Conference"
is [from-memory]: copy the booktitle from the IEEE Xplore page.

**Where to cite the new entries:** `xu2018feature` at l.170-171 (squeezing); `kurakin2017adversarial`
in the FGSM/PGD paragraph (S1); `croce2022evaluating`, `croce2020reliable` in Related Work
(adaptive evaluation, overestimation); `chhipa2024open`, `zhao2025one` in the gap sentence;
`zheng2025revisiting`, `jing2024pad` in patch defenses (PAD is the runnable modern patch baseline
[from-memory, code github.com/Lihua-Jing/PAD]). Also strengthen the existing SAC citation: SAC's
Tab. 1 (COCO, 75x75 patches) reports JPEG 39.7 -> 22.8 mAP and spatial smoothing 40.4 -> 23.2 mAP
from non-adaptive to BPDA attacks [searched https://ar5iv.labs.arxiv.org/html/2112.04532, read
by a summariser: confirm against the PDF before quoting]. That is the closest prior evidence
against the paper's recommendation and belongs in Related Work.

---

## 6. Claims to delete or soften

| ID | PDF line(s) | Claim | Action | Why |
|---|---|---|---|---|
| S1 | 080-081; 293-300; Fig. 2 caption | "single-step FGSM is harder to counter ... and explain the spectral mechanism"; "PGD ... high-frequency ... FGSM ... lower-frequency" | Delete from contributions; present as one hypothesis among three, or measure | No spectrum was measured. Rivals: (a) multi-step attacks transfer less than single-step ones (Kurakin et al.: "multi-step attack methods are somewhat less transferable than single-step attack methods" [searched https://arxiv.org/abs/1611.01236]), and oblivious defense is a transfer from model to model+transform; (b) recovery % divides by a larger PGD drop (YOLO 0.390 vs 0.247 mAP). To test (needs GPU): radially averaged power spectra of $\delta$ for FGSM vs PGD on a few hundred images |
| S2 | 162-163 | "PGD is strictly stronger than FGSM under the same budget" | "In every setting we ran, PGD reduced the metric more than FGSM" (R-E13) | Not guaranteed with one random start and 10 steps; true empirically here: 0.0619 < 0.2051, 0.2005 < 0.2488, 0.294 < 0.346 (Tab. 2) |
| S3 | 016; 045-047; 091; 165-167 | "physically realizable" | Delete (R-E4) | E4 |
| S4 | 021-022; 074-075; 108-110; 217; 342-344 | "classification-era defense rankings do not transfer to zero-shot detection" | Drop as a contribution. At most: "under our oblivious setting and parameters, JPEG ranked higher, and NLM lower, than their classification-era reputation" | The CAAD claim is wrong (E17); Guo's ranking was for classifiers trained on transformed images under gray/black-box attacks (E19); NLM and Random Resize here differ from the cited methods (E7, E8) |
| S5 | 022-025; 077-079; 305-307; 346-347 | "only defenses positive in all nine settings" | R-E2 | E2 |
| S6 | 279-282; 348-350 | "It is therefore our default recommendation for detection" | Make the recommendation conditional on RQ4 and clean cost | Recommending a deployment default from oblivious numbers, while Sec. 5 expects adaptive attacks to cut recovery "substantially", was the core objection |
| S7 | 269-270 | "TVM-based branches spread rather than remove the patch artifact" | "One possible reason is ..." | Not measured |
| S8 | 308-313 | Patch: "mild smoothing cannot remove it yet blurs the surrounding clean content" | Hypothesis; test with RQ1 | Negative solo recovery under Patch may be mostly clean cost (e.g. TVM on YOLO-Patch 0.4017 vs clean 0.4519); RQ1 separates the two |
| S9 | 314-319 | TVM helps YOLO but hurts Florence-2 because the decoder "relies on" fine texture | Hypothesis | Unexplained: TVM is +31.3% on Florence-PGD (Tab. 4) |
| S10 | 027-028; 289-291 | "The same configurations transfer across both architectures without modification" | "We used the same transform parameters for both models, without tuning" | "Transfer" implies a tested claim; the best defense differs by model (l.301-303) |
| S11 | 060-061 | "so the rankings are not an artifact of a small subset" | Delete or replace with CIs | Median/Gaussian change sign on YOLO-Patch between the N=1000 subset and N=5000 (E2; Gaussian -11.4% vs +1.4%) |
| S12 | 013; 054-055 | "upper-bounds robustness to an adaptive adversary" | Keep, but say "an upper bound on robustness against the strongest adaptive adversary" | Fine as a worst-case statement; any specific adaptive attack is not guaranteed stronger |

---

## 7. Checklist for the user and teammates

| # | Owner | Item | Blocks |
|---|---|---|---|
| 1 | Digvijay (Team table) | Import the YOLO N=5000 `summary_{fgsm,pgd,patch}.json` into `results/`; confirm the run used `archive/Phase3/run_survey_yolo.py` as committed | Citing Tab. 3 |
| 2 | Ankush | Import the OCR N=5000 summaries (`results_survey_florence_ocr/`) | Citing Tab. 5 |
| 3 | Digvijay | Check Random Resize 0.1311 under both FGSM and PGD (N11) | Tab. 3 |
| 4 | All three | **Ask whether the N=5000 `checkpoint_{attack}.pkl` files or `clean_*.json` dumps still exist** on the machines that ran them. They contain `clean+<def>` for every defense, so RQ1 at N=5000 for all 19 defenses and N=5000 bootstrap CIs could be done on CPU, without any GPU | RQ1/RQ2 at N=5000 |
| 5 | User | Get the APSCON camera-ready PDF (not in the repo or `~/Downloads/IMP Files/`; `docs/CV_Research_Paper___CVPR_Format.pdf` is the IITH course report, not APSCON). Diff text, figures and numbers against the revised draft; cite it in the third person; rerun rather than reuse its Florence FGSM/PGD results (TMLR forbids reusing archival results) | Dual-publication risk |
| 6 | User | Fill the `apscon2026` BibTeX placeholder | References |
| 7 | User | val2017: only 20 images are on disk and any real run needs all 5,000. Per the user (24 Oct), do not install the dataset on this laptop now; download it on the GPU machine (A6000 if confirmed) when runs start | All GPU runs |
| 8 | User | Check whether EmergingAD's OpenReview made the rejected PDF and reviews public, and paste the exact review text into `docs/` (only a summary exists on disk) | Anonymity at TMLR; rebuttal-style response |
| 9 | Anyone, CPU | Run `phase4/p4_rescore.py` (N=1000 dumps; no model, no images) for the first RQ1/RQ2 tables and Florence score-free metrics | RQ1/RQ2 first draft |
| 10 | Anyone | Confirm the pages/DOIs marked "confirm" in 5.2; check Guo et al.'s body for the word "weak"; confirm SAC Tab. 1 numbers | References |
| 11 | Anyone | Fix or drop Random Resize for detection (E8) | Tabs. 3-4 |
| 12 | User | Decide title scope: YOLO-World only for RQ4, unless the Florence-2 adaptive attack gets written | Title, abstract |
| 13 | Anyone | After a clean_control Florence OCR run: report the share of images with an empty clean OCR output (2.9) | OCR caveat |
| 14 | User | Apply Sec. 1 errata on Overleaf before any new submission, even a workshop one | Every submission |

Nothing here was committed. `docs/memory.md` "Phase-4 Status" says `docs/paper_revision.md` is in
progress; update that line when this file is accepted.
