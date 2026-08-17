# archive/Phase3 — read-only Phase-3 code

Moved here when Phase 4 started (2026-08-17). **Nothing in this folder should be
edited.** It is kept so every committed number stays traceable to the exact code
that produced it.

## What is here

| File(s) | Produced | Status |
|---|---|---|
| `{FGSM,PGD,Patch}_Phase3_YOLO_v2.ipynb` | the N=1000 YOLO paper rows | **frozen** — the notebooks hold the only stored outputs (~31 KB each, 11 executed cells) backing those numbers. Never strip their outputs. There is no `.py` equivalent. |
| `{FGSM,PGD,Patch}_Phase3_Florence_v2.py` | the N=1000 Florence detection rows | **frozen** (Jupytext superset of the deleted notebooks) |
| `{FGSM,PGD,Patch}_Florence2_OCR_Robust.py` | the N=1000 OCR rows | **frozen** |
| `run_survey_{yolo,florence_detection,florence_ocr}.py` | the N=5000 survey | superseded by `phase4/`, kept because the committed Florence-detection N=5000 summaries came from here |
| `run_parallel_survey.sh` | tmux launcher for the three survey runs | superseded by the Phase-4 entry points |

## What did NOT move, and why

`phase3_common.py` stays at the **repository root**. It is the shared defense /
NMS / checkpoint core that `phase4/` imports, and importing the *same* module is
the only reason Phase-4 numbers can be compared against the Phase-3 attacked
runs. Archiving it would have meant either duplicating it or breaking that
guarantee.

`results/` also stays where it is — the committed `summary*.json` are referenced
by path from `README.md` and `docs/memory.md`.

## If you ever need to re-run something in here

These scripts `import phase3_common`, which sits one level above this directory,
and they resolve the dataset relative to the current working directory. Python
puts the *script's* directory on `sys.path`, not the caller's, so run them from
the repo root with `PYTHONPATH` set:

```bash
cd /path/to/Adversarial_Attack_Defence_Paper
PYTHONPATH=. python archive/Phase3/run_survey_yolo.py \
  --image-dir ./val2017 --ann-file ./annotations/instances_val2017.json \
  --gpu 0 --num-images 5000 --tier survey --attacks fgsm pgd patch
```

Fixing the imports inside the files would mean editing frozen code, so the
`PYTHONPATH` prefix is the deliberate workaround. Two further gotchas that were
already true before the move:

- **They write to the OLD root paths.** `OUTPUT_DIR` in the three
  `*_Phase3_Florence_v2.py` files and in `run_survey_yolo.py:166` points at
  `./results_phase3_*` / an absolute `results_survey_yolo`, so a re-run recreates
  directories at the repo root instead of under `results/`.
- **They leave the per-condition COCO dumps on disk** (hundreds of MB per run).
  `.gitignore` keeps them out of git, but delete `results_*/[!s]*.json` before
  sharing a directory. Phase 4 fixed this — see `phase4/p4_logging.py`.
- **Dataset paths differ by track:** the YOLO and OCR scripts default to
  `./val2017`, the Florence detection script to `./Dataset/val2017`.
- **PGD alpha differs by track:** `PGD_Florence2_OCR_Robust.py` defaults to
  `--alpha 0.003` (eps/10), every other track uses eps/4 = 0.0075. See the
  README note; do not "fix" it, the committed OCR row depends on it.
