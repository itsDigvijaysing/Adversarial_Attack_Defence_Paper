# Handoff: Phase-4 runs on the A5000 server

Written 16 Nov 2026 on the laptop. It covers the code state, setup, run order, exact commands and
what to bring back. The paper-side map of every red `\pending{}` item is in
`docs/paper_draft/README.md` ("What is still red in the PDF"). This file follows it, with one
correction: `phase4/run_queue.py`, which that README and `main.tex` (line 736) name as the
launcher, **was never written**. Run the plain commands below instead.

## 1. State of the code (checked 16 Nov, CPU only)

| Check | Result |
|---|---|
| `py_compile` of `phase4/*.py`, `phase3_common.py`, `tests/*.py` | OK |
| `tests/test_offline_logic.py` | 38 / 38 |
| `tests/test_flow_stubbed.py` | 20 / 20 |
| `tests/test_attack_units.py` (vlm_ftune, CPU torch, fake detector) | 20 / 20 |
| `tests/test_rescore.py` | 20 / 20 |
| `--preflight`, both models | weights found; NOT READY only because the laptop has 20 of 5000 images (expected) |
| `scripts/setup_server.sh` | `bash -n` OK; conda activation works under its `set -euo pipefail` |

**Not yet verified:** the real YOLO-World model has not run the current attack code (union-max
loss, native attack space, true median gradient, `--pgd-select best`, batched EOT). It has only run
against the fake detector in the tests. The GPU smoke test in step 4 is that check: do not start
long runs before it passes.

**Before pushing from the laptop:** `.gitignore` has an unstaged change (`downloads/`, where the
setup script puts the ~1 GB of COCO zips). Stage it with this file:

```bash
git add .gitignore docs/HANDOFF_A5000.md
```

## 2. The A5000

- Ampere, sm_86, 24 GB. torch 2.6.0+cu118 supports it (sm_50 to sm_90); the driver needs CUDA 11.8
  support (any driver from 520 on).
- `--eot-batch auto` turns batching on here (24 GB is at least 16 GiB). Each run's summary records
  the choice under `execution.eot_batch`. On an out-of-memory error it falls back to one branch at
  a time for the rest of the run and records that in `eot_batch_oom_fallback`. The numbers are the
  same either way.
- Measured peaks on the 6 GB laptop GPU were ~2.3 GB (YOLO) and ~1.3 GB (Florence-2). So 2 or 3
  jobs can share the card; see step 7.

## 3. Setup (once, needs internet)

```bash
git clone git@github.com:itsDigvijaysing/Adversarial_Attack_Defence_Paper.git
cd Adversarial_Attack_Defence_Paper
bash scripts/setup_server.sh      # Miniforge if no conda, env, COCO val2017, weights, preflight, tests
```

- It is safe to re-run: each step skips what is already there.
- Disk: ~10 GB (env ~7, COCO ~1.6, weights ~1).
- It ends with both preflights and the three test suites. Every line should say READY or PASS.

In every new shell or tmux pane:

```bash
conda activate vlm_ftune && cd <repo> && export HF_HOME=$PWD/weights/hf
export HF_HUB_OFFLINE=1           # only if the GPU node has no internet
```

## 4. Gate 0: GPU smoke tests (2 images each, a few minutes)

```bash
python phase4/p4_yolo.py --experiment adaptive_bpda --attack-space native --pgd-select best \
    --smoke-test --device cuda --gpu 0
python phase4/p4_yolo.py --experiment clean_control --smoke-test --device cuda --gpu 0
python phase4/p4_florence.py --experiment clean_control --smoke-test --device cuda --gpu 0
```

Pass means:

- all three exit 0;
- each prints `[out] .../smoke_test/summary_..._latest.json`;
- no NaN appears in the summaries;
- the adaptive one shows `execution.eot_batch` = on.

The smoke numbers are not results and are never committed: `smoke_test/` is gitignored.

## 5. Gate 1: timing pilot (~10 min to an hour)

The image count is part of the checkpoint signature, so a pilot cannot be resumed by the real run.
Give it its own folder:

```bash
A="--experiment adaptive_bpda --attack-space native --pgd-select best"
python phase4/p4_yolo.py $A --n 100 --pgd-iters 50 --alpha 0.002 \
    --target ens_jpeg_median_gaussian --seed 1 --gpu 0 \
    --output-dir results/phase4/pilot_timing
```

The heartbeat prints `s/img`. Multiply it by the image counts below to plan the queue.

| Job | Images | Relative cost per image |
|---|---|---|
| Median | 1,000 | 1 defense branch, cheapest |
| EnsJMG | 1,000 | 3 branches |
| Ens4W | 1,000 | 4 branches including TVM (200 inner iterations per forward), slowest |

**Unmeasured guess, for booking only:** 10 to 15 s per image for a 50-step EnsJMG job on an
A5000. That is ~3 to 4 h per 1,000-image job, and ~40 to 50 GPU-hours for the whole plan run one
job at a time. Trust the pilot over this. Delete `results/phase4/pilot_timing/` afterwards, or
leave it uncommitted.

## 6. The runs, in order

Common flags: `A` as above. `K=50` and `alpha=0.002` are an untested suggestion; J1 decides K.

**J1: convergence (decides K).** n=300, EnsJMG, seed 1.

```bash
python phase4/p4_yolo.py $A --n 300 --target ens_jpeg_median_gaussian --seed 1 \
    --pgd-iters 50  --alpha 0.002 --gpu 0 --output-dir results/phase4/adaptive_bpda_yolo_iters50_n300
python phase4/p4_yolo.py $A --n 300 --target ens_jpeg_median_gaussian --seed 1 \
    --pgd-iters 100 --alpha 0.002 --gpu 0 --output-dir results/phase4/adaptive_bpda_yolo_iters100_n300
```

Compare `headline.adaptive_attacked_defended_mAP` in the two `_latest` summaries.

- If 100 steps gives a noticeably lower value, use `ITERS=100` for J2 to J5. "Noticeably" is a
  judgement call; my suggestion is more than 0.005 mAP.
- Otherwise use `ITERS=50`.

**J2: main adaptive runs (9 jobs, Table 5).**

```bash
ITERS=50     # or 100, from J1
for T in median ens_jpeg_median_gaussian ens_4way; do
  for S in 1 2 3; do
    python phase4/p4_yolo.py $A --n 1000 --pgd-iters $ITERS --alpha 0.002 \
        --target $T --seed $S --gpu 0 --log-name run_${T}_s${S}.log
  done
done
```

All 9 jobs share `results/phase4/adaptive_bpda_yolo/`. Each keeps its own checkpoint and its own
`summary_adaptive_bpda_yolo_<target>_seed<S>..._latest.json`. Table 5's five columns are the
`headline` fields, averaged over the 3 seeds:

- `clean_defended_mAP`
- `oblivious_attacked_undefended_mAP`
- `oblivious_attacked_defended_mAP`
- `adaptive_attacked_undefended_mAP`
- `adaptive_attacked_defended_mAP`

Check (i) is `surrogate_objective.frac_images_adaptive_below_oblivious`.

**J3: larger budget, check (iii).** Alpha is scaled with eps.

```bash
python phase4/p4_yolo.py $A --n 300 --target ens_jpeg_median_gaussian --seed 1 \
    --pgd-iters $ITERS --eps 0.1 --alpha 0.0067 --gpu 0 \
    --output-dir results/phase4/adaptive_bpda_yolo_eps0.1
```

**J4: max vs mean objective, check (iv).** J5: true median gradient vs identity, check (v).
These write to the same folder as J2 under their own summary names.

```bash
python phase4/p4_yolo.py $A --n 1000 --pgd-iters $ITERS --alpha 0.002 \
    --target ens_jpeg_median_gaussian --seed 1 --eot-loss mean --gpu 0 --log-name run_j4_mean.log
python phase4/p4_yolo.py $A --n 1000 --pgd-iters $ITERS --alpha 0.002 \
    --target median --seed 1 --median-surrogate identity --gpu 0 --log-name run_j5_identity.log
```

**J6: clean cost of every transform (Sec. 4.2).** This does not depend on J1, so it can run at
any time, e.g. in a second pane.

```bash
python phase4/p4_yolo.py     --experiment clean_control --n 5000 --gpu 0
python phase4/p4_florence.py --experiment clean_control --n 1000 --gpu 0   # detection + OCR
```

## 7. Running side by side on one card

- Run 2 or 3 jobs at once in separate tmux panes, all with `--gpu 0`. Watch `nvidia-smi`. If
  utilisation is already near 100 % with one job, more jobs add nothing.
- Every concurrent job in the same folder needs its own `--log-name` (the commands above have
  one).
- **Never run the same command twice at once:** identical flags write the same checkpoint file.
- A natural split:
  - pane 1: J1, then J2;
  - pane 2: J6, then J4, J5 and J3 once K is known.

## 8. Crashes and resume

- Re-run the identical command. The checkpoint is keyed by the full config, so the job continues
  where it stopped.
- `seconds_per_image` counts only images computed in that session, so a resumed run's rate is
  honest.
- Never resume after changing a flag: that starts a new config, by design.

Troubleshooting:

| Symptom | Fix |
|---|---|
| Out of memory | Batching falls back automatically. If it persists, use `--eot-batch off` or run fewer jobs at once |
| `NOT READY ... SHORT` | `val2017/` has fewer than 5000 images: re-run `setup_server.sh` |
| Hub errors on a node with no internet | `export HF_HUB_OFFLINE=1`; the weights are already in `weights/hf/` |
| A suspicious mAP | Re-run with `--verbose-eval` for the full pycocotools output |

## 9. What to bring back

Copy the whole `results/phase4/` folder, not just what git tracks. From the laptop:

```bash
rsync -av <server>:<repo>/results/phase4/ results/phase4/
```

| File | In git? | Keep it? |
|---|---|---|
| `summary_*.json`, `summary_*.md` (`_latest` + timestamped) | yes | these are the citable results |
| `run*.log` | yes | |
| `per_image_*.json` | no | keep |
| `checkpoint_*.pkl` | no | keep: per-image results, needed later for seed/restart worst-case aggregation or paired intervals on Table 5 |

Do not commit `pilot_timing/` or any `smoke_test/` folder. The COCO dumps are deleted
automatically by the artifact policy; that is intended.

## 10. Known gaps (none block the runs)

- **The launcher does not exist.** `phase4/run_queue.py` is named in `docs/paper_draft/README.md`
  (lines 74, 82, 103) and in `main.tex` line 736 ("scheduled across GPUs by
  phase4/run_queue.py"). Fix that sentence before the paper goes out, or write the launcher.
- **Stale test counts.** README "Phase 4, how to run" still says 37 offline and 14 attack checks.
  Now 38 and 20.
- **Adaptive attack is YOLO only.** Adaptive Florence-2 is not implemented. If it is not added,
  scope the title to YOLO-World.
- **N=5000 summaries missing.** The YOLO (Digvijay) and OCR (Ankush) N=5000 summaries are still
  not in the repo.
