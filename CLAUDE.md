# CLAUDE.md — Adversarial Attack/Defence Survey

Guidance for AI assistants working in this repo. See `README.md` (usage) and `docs/memory.md`
(project state, results, citations). Phase-1/2 history is in `docs/memory_phase2_archive.md`.

## What this project is
A survey of input-transform adversarial defenses for two open-vocabulary models —
**Florence-2-base** (VLM: object detection + OCR) and **YOLOv8x-worldv2** (object detector) — on
**COCO val2017**, under **FGSM / PGD / Patch** attacks. Two evaluation tracks: **detection**
(COCO mAP@[.5:.95]) and **OCR** (text self-consistency). Scope explicitly includes **negative
results** and **novel 2025–2026 methods**.

## Environment & run
- `conda activate vlm_ftune` (PyTorch 2.6.0+cu118, transformers 4.51.0, see `environment.yml`).
- Models: `microsoft/Florence-2-base` revision `refs/pr/26` (float16); `yolov8x-worldv2.pt`
  (ultralytics auto-downloads).
- Data (gitignored, fetch separately): COCO `val2017/` + `annotations/instances_val2017.json`.
- **Phase-4 runs (current):** `python phase4/p4_yolo.py --experiment {clean_control,adaptive_bpda}`
  and `python phase4/p4_florence.py --experiment clean_control [--tracks florence_det florence_ocr]`.
  See README for the full CLI. Phase-3 survey runs are archived (`archive/Phase3/`, needs
  `PYTHONPATH=.`).
- **Do NOT execute training/eval scripts unless explicitly asked** — they need GPU + model + COCO.
  For code changes, verify with `python -m py_compile <file>` **and**
  `python tests/test_offline_logic.py` (30 torch-free checks: CLI, output paths, seeded selection,
  OCR helpers, artifact policy, and an AST check that p4's 7 ensembles match `phase3_common`).

## Layout (restructured for Phase 4 on 2026-08-17)
- **`phase3_common.py`** — shared core, **stays at the repo root**: GPU defenses (`tvm_gpu`,
  `median_gpu`, `gaussian_gpu`, `svd_gpu`, `bilateral_gpu`), CPU defenses (`jpeg_cpu`, `nlm_cpu`,
  `bit_depth_cpu`, `random_resize_pad_cpu`, `dithering_cpu`, `anisotropic_diffusion_cpu`, optional
  `bm3d_cpu`), `apply_all_defenses_gpu` (5 locked solos) / `apply_survey_defenses` (full survey
  bank), `merge_branches_nms`, `assemble_results`, COCO eval, `SurveyCheckpoint` +
  `config_signature`. `phase4/` imports it; importing the SAME module is what makes Phase-4 numbers
  comparable to the Phase-3 attacked runs, so do not move or fork it.
- **`phase4/`** — the only active code. Two entry points, one per model, plus shared logic:
  - `p4_core.py` — everything not model-specific, written ONCE: CLI (`base_parser`/`finalize_args`),
    path resolution, seeded `select_images`, `build_branches`, `run_per_image` (checkpointed loop +
    hard asserts), `score_detection`/`score_ocr`, timestamped `write_summary`/`write_markdown`,
    and the `clean_control` experiment shared by both models.
  - `p4_yolo.py` — ENTRY POINT. Letterbox, detection parsing, the confidence-suppression attack
    objective, oblivious PGD, and the BPDA+EOT adaptive attack (`adaptive_bpda`, YOLO-only).
  - `p4_florence.py` — ENTRY POINT. Florence-2 loaded ONCE serving both `<OD>` and `<OCR>`;
    `FLORENCE_TO_COCO`, the fabricated `compute_score`, NMS. Attacks NOT ported yet (they live in
    `archive/Phase3/`) — port them only when a Phase-4 attack experiment is defined.
  - `p4_logging.py` — the run-artifact policy: `run_log` (tee to `run.log`), `eval_dump_dir`
    (COCO dumps to a temp dir, deleted), `artifact_report` (committed vs local split).
- **`tests/test_offline_logic.py`** — 30 torch-free checks; run it after any `phase4/` edit.
- **`docs/`** — report + presentation PDFs, `memory.md`, `memory_phase2_archive.md`, `GPU`,
  `tmux.txt`. **`figures/`** — result images. **`results/`** — Phase-3 output dirs plus
  `results/phase4/<experiment>_<model>/` for new runs.
- **`archive/`** — read-only history. **`archive/Phase3/`** now holds the 9 locked v2
  scripts/notebooks and the 3 N=5000 survey scripts — **read `archive/Phase3/README.md` before
  touching them.** Also `Scripts_Extra/`, `Logs_Extra/` (3.2 GB of historical dumps, gitignored),
  `Very_OLD/`. No active code reads from `archive/`.

## Phase-4 conventions
- **One process = one model = one GPU = one tmux pane.** Prefer `.py` entry points over notebooks:
  a notebook cannot be detached, resumed, or diffed usefully. The archived YOLO v2 notebooks are the
  exception — they store the only outputs backing those numbers.
- **Artifact policy (`p4_logging.py`):** every run leaves exactly three shareable files —
  `summary_*.json`, `summary_*.md`, `run.log` — all a few KB. The per-condition COCO dumps go to a
  temp dir and are deleted (`--keep-eval-dumps` overrides). Never reintroduce dumps into a results
  directory; that is what made Phase-3 run dirs unshareable.
- **Summaries are timestamped, the directory is not.** `summary_<exp>_<model>_n<N>_<ts>.json` never
  overwrites, `_latest.json`/`_latest.md` mirrors the newest run for stable citation, and the fixed
  run directory keeps the checkpoint resumable. Do not switch to per-run directories — that breaks
  resume, so a crashed 5000-image run would restart from zero.
- **Timing:** `seconds_per_image` divides by images *computed this run*, never by `len(files)` —
  checkpoint-resumed images cost nothing and would fake the rate.
- **Every filename written into `results/` must start with `summary`** to survive
  `.gitignore`'s `!results/**/summary*.json` negation.

## Critical conventions & gotchas
- **Never modify the frozen v2 files or `pc.SOLO_DEFENSES`/`pc.ENSEMBLES`.** v2 scripts iterate
  those 5 solos / 3 ensembles and `apply_all_defenses_gpu` produces exactly those keys — adding a
  name (e.g. `svd`) KeyErrors every v2 run. Survey defenses live in `TIER*/SURVEY_*` constants.
- **Florence detection scores are fabricated** (`_compute_score`: geometry, not confidence) because
  Florence `<OD>` emits no scores. Applied identically to all conditions → deltas fair, absolute mAP
  geometry-ranked. YOLO uses real confidences. Disclose this in the paper.
- **OCR metric = self-consistency**, not accuracy: char-level `SequenceMatcher` vs the model's *own
  clean* OCR output (no COCO OCR GT). `clean_baseline` ≈ 1.0 by construction. Never call it
  "word-recovery ratio."
- **Attack pixel spaces differ:** Florence attacks in **normalized** (ImageNet) space, clamp
  [−2.5,2.5]; YOLO in **raw [0,1]**.
- **Checkpointing:** use the config-keyed `SurveyCheckpoint`, never the legacy `DetectionCheckpoint`
  with bare `has()`/skip — that caused the documented v1 cross-run contamination. New survey runs
  derive `defense_names` from the producer's returned keys (bm3d may be absent).
- **Cite only committed `summary.json` numbers.** Older PGD-YOLO figures in past notes were stale
  (no committed file backs them); correct v2 PGD = 0.4765→0.0737→0.4128 (ens_blur_tvm_combo).
- **Dataset paths:** Phase 4 auto-resolves (`./val2017`, then `./Dataset/val2017`, ...) so both
  models score the same images. The archived Phase-3 scripts are inconsistent by track — YOLO/OCR
  default to `./val2017`, Florence detection to `./Dataset/val2017`.
- **PGD alpha is not uniform in Phase 3:** eps/4 = 0.0075 everywhere except
  `archive/Phase3/PGD_Florence2_OCR_Robust.py:642`, which uses eps/10 = 0.003 — and that is the
  value behind the committed N=1000 OCR PGD row. Never write "alpha = eps/4" as a blanket claim.
- **eps=0.03 is NOT the same budget for both models.** Florence adds it in normalized space
  (≈0.03x0.225 ≈ 1.7/255 in pixel units); YOLO adds it in raw [0,1] (7.65/255). ~4.4x apart, which
  plausibly explains why YOLO PGD collapses further than Florence PGD.
- **The YOLO attack objective is NOT cross-entropy.** It maximizes `-sum_anchors max_class
  pred[:,4:,:]` (post-sigmoid confidences; no box term, no objectness, no self-labels). Only
  Florence uses token-level CE against its own clean decode. See `docs/memory.md`.
- **Novel-method citations are verified but mostly NOT evaluated on Florence-2** (only EigenShield
  is). Cite preprints as preprints; see the table in `docs/memory.md`.

## Git
- **Do not commit unless explicitly asked.** Per the user's global preference, **never add
  `Co-Authored-By` or any AI/assistant attribution** to commits/PRs.
- Only `summary*.json`, `run.log`, `.png`, code, and docs are tracked under `results/`; dump JSONs,
  `*.pkl`, and survey checkpoints are gitignored (702 MB of dumps sit behind those rules).
- **Never write `results/*` in `.gitignore`** — excluding a directory makes every `!` negation
  inside it unrescuable, which silently drops all committed summaries. Match files
  (`results/**/*.json` + negations), not directories.
- **Do not create branches either** unless explicitly asked. A task description that says "commit
  the scripts" is not permission — only the user's own instruction is.
- **`results/**/smoke_test/` is ignored wholesale** so a `--smoke-test` plumbing check can never be
  committed as a result.
- **The archived Phase-3 scripts still write to the OLD root paths.** `OUTPUT_DIR` in the three
  `archive/Phase3/*_Phase3_Florence_v2.py` files and `archive/Phase3/run_survey_yolo.py:166` points
  at `./results_phase3_*` / an absolute `results_survey_yolo`, so a re-run recreates dirs at the
  repo root instead of under `results/`. They are frozen; do not repoint them — Phase-4 code writes
  to `results/phase4/` correctly.
- **N=5000 provenance:** only the Florence-2 detection summaries are committed here. The YOLO
  (Digvijay) and OCR (Ankush) N=5000 runs happened on their machines — real, confirmed numbers whose
  files are not yet in this repo. Never call them unverified; the fix is to import the files.

## graphify (knowledge graph — `graphify-out/`)
Built 2026-08-09 over all 96 files: 1,027 nodes / 1,856 edges / 29 communities. See
`graphify-out/README.md` for the full file map and provenance.
- **Before answering architecture or cross-file questions,** read `graphify-out/GRAPH_REPORT.md`
  (god nodes, communities, AMBIGUOUS edges) instead of grepping blind. For "how does X relate to
  Y", prefer `graphify query "<q>"` / `path "<A>" "<B>"` / `explain "<concept>"` — these traverse
  EXTRACTED + INFERRED edges rather than scanning files.
- **Keeping it current:** `graphify update .` after code edits (AST-only, no API cost).
  `/graphify . --update` only when `.md`/`.pdf`/images changed — that path dispatches extraction
  subagents and costs tokens. The content-hash cache skips unchanged files either way.
- **Never relocate `graphify-out/`** — the name is hardcoded in the graphify package
  (`cache.py:48`, `detect.py:19`) and in `~/.claude/skills/graphify`. Moving it makes the next run
  recreate it at root *and* ingest the moved copy as source corpus.
- **Tracked:** `GRAPH_REPORT.md` + `README.md` only. `cache/`, `graph.json`, `graph.html`,
  `manifest.json`, `cost.json` are gitignored — regenerable or machine-local. Don't delete
  `cache/`; that is what makes re-extraction free.
- **Figure-derived nodes are only as good as the figure labels.** Vision extraction flagged two
  result figures as contradicting `docs/memory.md` — treat AMBIGUOUS edges as questions, not facts.
- `.claude/settings.json` holds a PreToolUse hook (from `graphify claude install`) that reminds
  sessions the graph exists before Glob/Grep. It self-guards on `graph.json` being present.
