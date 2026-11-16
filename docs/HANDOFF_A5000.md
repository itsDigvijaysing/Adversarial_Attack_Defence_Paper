# Handoff: Phase-4 runs on the GPU server

Rewritten 17 Nov 2026 on the laptop. It replaces the 16 Nov version. The server is an RTX 5000 Ada
(32 GB), repo at `~/m_work/Adversarial_Attack_Defence_Paper`, already set up on 16 Nov (environment,
COCO val2017, YOLO-World, CLIP and Florence-2 weights; both preflights READY).

**Read this first.** Three things in this commit have never run on a real model or a GPU. They
were written on the laptop, where only fake models are available:

| New code | File | First real check |
|---|---|---|
| Adaptive attack on Florence-2 | `phase4/p4_florence.py --experiment adaptive_bpda` | Gate 0, command 2 |
| Third model, SmolVLM-256M yes/no questions | `phase4/p4_smolvlm.py` | Gate 0, command 3 |
| `--deterministic` (repeatable GPU runs) | all three scripts | Gate 1 |

Expect to fix small errors in them at Gate 0. Do not start a long run of a script before its Gate 0
command exits 0. The YOLO-World attack is unchanged in what it computes and already passed on this
server on 16 Nov.

There is no launcher script. Every job is one plain command.

## 1. Once, after `git pull`

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlm_ftune
cd ~/m_work/Adversarial_Attack_Defence_Paper && export HF_HOME=$PWD/weights/hf
# the third model's weights (about 1 GB, needs internet; HF_HUB_OFFLINE must be unset here)
python -c "from huggingface_hub import snapshot_download as d; print(d('HuggingFaceTB/SmolVLM-256M-Instruct'))"
python tests/test_offline_logic.py | tail -2          # 38 passed
python tests/test_flow_stubbed.py  | tail -2          # 21 passed
CUDA_VISIBLE_DEVICES='' python tests/test_attack_units.py | tail -2   # 23 passed
python phase4/p4_smolvlm.py --preflight
```

All of these passed on the laptop on 17 Nov with the same counts. If one fails here, the
environment differs: fix that before anything else.

In every new shell or tmux pane:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlm_ftune
cd ~/m_work/Adversarial_Attack_Defence_Paper && export HF_HOME=$PWD/weights/hf
export HF_HUB_OFFLINE=1      # after the SmolVLM download: keeps Florence-2's remote code fixed
```

## 2. Gate 0: smoke tests (2 images each, minutes)

```bash
python phase4/p4_yolo.py --experiment adaptive_bpda --attack-space native --pgd-select best \
    --deterministic --smoke-test --device cuda --gpu 0
python phase4/p4_florence.py --experiment adaptive_bpda --pgd-select best \
    --deterministic --smoke-test --full-gen --device cuda --gpu 0
python phase4/p4_smolvlm.py --deterministic --smoke-test --device cuda --gpu 0
python phase4/p4_yolo.py --experiment clean_control --all-transforms --smoke-test --device cuda --gpu 0
python phase4/p4_florence.py --experiment clean_control --all-transforms --smoke-test --device cuda --gpu 0
```

Pass means: exit 0, a `[out] .../smoke_test/summary_..._latest.json` line, no NaN in the summary.
Also check:

- Florence-2: the `[guard] differentiable preprocessing vs the processor` line. A value under
  about 0.1 is fine (it is in normalized units; 1/255 is about 0.017). The smoke test uses 2
  attack steps, so `surrogate_objective.frac_images_adaptive_below_oblivious` means nothing
  there. Judge it on Job 3: if it is well below 1.0 after 50 steps with the default
  `--eot-loss max` (which moves one branch per step), compare with `--eot-loss mean` on 100
  images and use whichever gives the lower defended mAP, stating which in the paper.
- SmolVLM: the `[guard] processor == normalize` line must print. If the script stops there with
  "Refusing to run", the processor still resizes or splits the image: fix `SmolAdapter.load`
  (the `size` / `do_image_splitting` arguments) before going on. Clean accuracy on 2 images means
  nothing; only check that it runs.

## 3. Gate 1: are repeated runs identical?

On 16 Nov the same 10-image YOLO command gave different attacked mAP twice (0.095 against 0.064
undefended). `--deterministic` asks torch for deterministic kernels; operations that have none
only warn. Measure it:

```bash
A="--experiment adaptive_bpda --attack-space native --pgd-select best --deterministic"
for R in a b; do
  python phase4/p4_yolo.py $A --n 10 --pgd-iters 50 --alpha 0.002 --seed 1 --gpu 0 \
      --target ens_jpeg_median_gaussian --output-dir results/phase4/pilot_determinism_$R
done
grep -h '"oblivious_attacked\|"adaptive_attacked' results/phase4/pilot_determinism_*/summary_*_latest.json
```

- **Identical:** keep `--deterministic` on every job below. Also compare the `s/img` line with the
  16 Nov rate (6.6 s per image for this target); if it is much slower, weigh that.
- **Still different:** drop the flag (it only costs time), and tell the paper side: the sentence
  on repeatability in the appendix is already worded for this case.

Delete both `pilot_determinism_*` folders afterwards, or leave them uncommitted.

## 4. The jobs

Measured on this server on 16 Nov (50 steps, one job alone): Median 4.3 s per image, the
three-filter ensemble 6.6, the four-filter ensemble 11.8, YOLO clean cost 0.6, Florence clean cost
3.6. **Run 3 jobs at once** in separate tmux panes, all with `--gpu 0`: 1.77 times the throughput
of one job; a fourth adds nothing. The Florence-2 adaptive job and the SmolVLM job have **not**
been timed: read their first heartbeat line (`s/img`) and multiply.

Common flags for the adaptive jobs (`D` is `--deterministic` or empty, from Gate 1):

```bash
D="--deterministic"
A="--experiment adaptive_bpda --attack-space native --pgd-select best $D"
F="--experiment adaptive_bpda --pgd-select best $D"       # Florence-2 (always native)
```

**Job 1, step-count check (YOLO-World, 300 images).** Decides `ITERS` for every later job.

```bash
python phase4/p4_yolo.py $A --n 300 --target ens_jpeg_median_gaussian --seed 1 \
    --pgd-iters 50  --alpha 0.002 --gpu 0 --output-dir results/phase4/adaptive_bpda_yolo_iters50_n300
python phase4/p4_yolo.py $A --n 300 --target ens_jpeg_median_gaussian --seed 1 \
    --pgd-iters 100 --alpha 0.002 --gpu 0 --output-dir results/phase4/adaptive_bpda_yolo_iters100_n300
```

Compare `headline.adaptive_attacked_defended_mAP`. If 100 steps is lower by more than about 0.005
mAP use `ITERS=100`, otherwise `ITERS=50`.

**Job 2, main YOLO-World table (9 runs, 1,000 images each).** Start the three-filter ensemble
and Median first; the four-filter ensemble is the slowest.

```bash
ITERS=50
for T in ens_jpeg_median_gaussian median ens_4way; do
  for S in 1 2 3; do
    python phase4/p4_yolo.py $A --n 1000 --pgd-iters $ITERS --alpha 0.002 \
        --target $T --seed $S --gpu 0 --log-name run_${T}_s${S}.log
  done
done
```

**Job 3, main Florence-2 table (new code).** Seed 1 for both targets first; seeds 2 and 3 only
if time allows.

```bash
for T in ens_jpeg_median_gaussian median; do
  python phase4/p4_florence.py $F --n 1000 --pgd-iters $ITERS --alpha 0.002 \
      --target $T --seed 1 --gpu 0 --log-name run_${T}_s1.log
done
```

**Job 4, clean cost of every transform.** Independent of Job 1; start it at any time.

```bash
python phase4/p4_yolo.py     --experiment clean_control --all-transforms --n 5000 --gpu 0
python phase4/p4_florence.py --experiment clean_control --all-transforms --n 1000 --gpu 0
```

Non-local means and BM3D run on the CPU and add seconds per image.

**Job 5, the third model.** 1,000 images (user, 17 Nov). A different `--n` is a new run, it
does not extend the old one.

```bash
python phase4/p4_smolvlm.py $D --n 1000 --gpu 0
```

Look at the clean accuracy in the first summary. If it is close to 0.5 (guessing), the 256M model
cannot do the task: run the same command with `--model HuggingFaceTB/SmolVLM-500M-Instruct
--output-dir results/phase4/presence_qa_smolvlm500` (download it first) and report both.

**Job 6, the three attack checks (YOLO-World).**

```bash
# a much larger budget should drive the defended score towards zero
python phase4/p4_yolo.py $A --n 300 --target ens_jpeg_median_gaussian --seed 1 \
    --pgd-iters $ITERS --eps 0.1 --alpha 0.0067 --gpu 0 --output-dir results/phase4/adaptive_bpda_yolo_eps0.1
# attack the most confident branch (default) against the average over branches
python phase4/p4_yolo.py $A --n 1000 --pgd-iters $ITERS --alpha 0.002 \
    --target ens_jpeg_median_gaussian --seed 1 --eot-loss mean --gpu 0 --log-name run_mean.log
# true median gradient (default) against a pass-through
python phase4/p4_yolo.py $A --n 1000 --pgd-iters $ITERS --alpha 0.002 \
    --target median --seed 1 --median-surrogate identity --gpu 0 --log-name run_identity.log
```

**If time runs short, this is the order of importance:** Job 2 (three-filter ensemble and Median,
seed 1, then seeds 2 and 3), Job 4, Job 5, Job 3, Job 2's four-filter ensemble, Job 6, Job 1's
100-step run.

A suggested split over three panes: pane 1 Job 1 then Job 2; pane 2 Job 4 then Job 3; pane 3
Job 5 then Job 6.

## 5. Rules while running

- Check the first heartbeat of every job (it prints after 25 or 50 images) before leaving it.
- Every concurrent job in the same folder needs its own `--log-name` (given above).
- Never run the same command twice at once: identical flags write the same checkpoint file.
- After a crash, re-run the identical command: it resumes. Changing any flag starts a new run.
- Out of memory: batching falls back by itself; otherwise run fewer jobs at once.
- A suspicious mAP: add `--verbose-eval`.

## 6. What to bring back

```bash
rsync -av <server>:~/m_work/Adversarial_Attack_Defence_Paper/results/phase4/ results/phase4/
```

Copy the whole folder, including the gitignored `per_image_*.json` and `checkpoint_*.pkl` (needed
for intervals later). Do not commit `pilot_*` or any `smoke_test/` folder. Then, on the paper
side:

```bash
python docs/paper_draft/make_adaptive_table.py --n 1000 --iters $ITERS   # tab_adaptive_{yolo,florence}.tex
```

It prints which summaries it used and skipped. The remaining red items in `main.tex` are filled by
hand from the summaries; `docs/paper_draft/README.md` maps each one to its job.

## 7. State of the code (17 Nov, laptop)

- All four suites pass on the laptop (processor only): offline 38, stubbed flow 21, attack units
  23, rescore 20. Every file compiles.
- Both new drivers (`p4_florence.adaptive_bpda`, `p4_smolvlm.presence_qa`) also ran end to end
  against FAKE models in a scratch script, including their summaries and tables. That proves the
  plumbing, not the behaviour on the real models: the real processors, half precision, beam
  search and the SmolVLM image layout are first exercised at Gate 0.
- `make_adaptive_table.py` was checked on synthetic summaries only.
- **Real-model check on the laptop GPU (RTX 3060, 17 Nov):** the Gate 0 YOLO-World command
  (`--attack-space native --pgd-select best --deterministic --smoke-test --device cuda`) ran
  twice, exit 0 both times, and all six headline mAP values were identical to 6 decimals
  (2 images, 2 attack steps). So the refactored command-line block and `--deterministic` work on
  the real model. torch warned that two operations have no deterministic kernel (the median with
  indices, and the backward of reflection padding), so a 50-step run can still drift: Gate 1 on
  the server remains the real test. Florence-2 and SmolVLM are not on the laptop and were not
  run on a real model.
- The scored defenses, the YOLO-World objective and its attack loop are untouched. In
  `p4_yolo.py` only the command-line block was moved into `add_attack_arguments` so Florence-2 can
  share it.

## 8. Known gaps

- No adaptive attack on the third model or on the Florence-2 OCR track.
- The Florence-2 adaptive numbers use a pixel-space budget (7/255), so they are not comparable
  with the older Florence-2 tables (about 1.7/255); the paper says so.
- The 5,000-image YOLO-World and OCR summary files do not exist any more (user, 17 Nov); those
  two tables stay as parsed from the earlier paper.
