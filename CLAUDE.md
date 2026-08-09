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
- Survey runs: `python run_survey_{yolo,florence_detection,florence_ocr}.py --tier survey
  --num-images 5000 --attacks fgsm pgd patch` (see README for full CLI).
- **Do NOT execute training/eval scripts unless explicitly asked** — they need GPU + model + COCO.
  For code changes, verify with `python -m py_compile <file>` (syntax only).

## Layout
- **`phase3_common.py`** — shared core: GPU defenses (`tvm_gpu`, `median_gpu`, `gaussian_gpu`,
  `svd_gpu`, `bilateral_gpu`), CPU defenses (`jpeg_cpu`, `nlm_cpu`, `bit_depth_cpu`,
  `random_resize_pad_cpu`, `dithering_cpu`, `anisotropic_diffusion_cpu`, optional `bm3d_cpu`),
  `apply_all_defenses_gpu` (5 locked solos) / `apply_survey_defenses` (full survey bank),
  `merge_branches_nms`, `assemble_results`, COCO eval, and `SurveyCheckpoint` + `config_signature`.
- **Locked v2 (N=1000, paper results) — DO NOT MODIFY:** `{FGSM,PGD,Patch}_Phase3_Florence_v2.py`,
  `{FGSM,PGD,Patch}_Phase3_YOLO_v2.ipynb`, `{FGSM,PGD,Patch}_Florence2_OCR_Robust.py`.
  The Florence `.ipynb` copies were deleted 2026-08-09 (unexecuted; the `.py` is a Jupytext
  superset carrying every code line plus the markdown as `# %%` comments). The YOLO v2 track is
  **notebook-only — there is no `.py`** — and those notebooks hold the only stored outputs
  (~31 KB each, 11 executed cells) backing the YOLO numbers. Never strip their outputs.
- **Survey scripts (N=5000):** `run_survey_{yolo,florence_detection,florence_ocr}.py`.
- **`docs/`** — report + presentation PDFs, `memory.md`, `memory_phase2_archive.md`, `GPU`,
  `tmux.txt`. **`figures/`** — result images. **`results/`** — all 13 run-output dirs, consolidated
  from the root `results_*` layout on 2026-08-09.
- **`archive/`** — read-only history: `Scripts_Extra/`, `Logs_Extra/` (3.2 GB of historical dumps,
  gitignored), `Very_OLD/`. No active code reads from `archive/`.

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
- **Dataset paths are inconsistent by track:** YOLO/OCR use `./val2017`; Florence detection uses
  `./Dataset/val2017`. Match the track you're editing.
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
- **Scripts still write to the OLD root paths.** `OUTPUT_DIR` in the three
  `*_Phase3_Florence_v2.py` files and `run_survey_yolo.py:166` points at `./results_phase3_*` /
  an absolute `results_survey_yolo`, so any re-run recreates dirs at the repo root instead of
  under `results/`. Repoint them when the post-review rewrite happens.

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
