#!/usr/bin/env python3
"""Build the adaptive-attack tables of the draft from the Phase-4 summaries.

    python docs/paper_draft/make_adaptive_table.py            # both models
    python docs/paper_draft/make_adaptive_table.py --model yolo --iters 50

Reads results/phase4/adaptive_bpda_<model>/summary_adaptive_bpda_<model>_
<target>_seed<S>*_latest.json, keeps the runs whose flags are the paper's
(--eot-loss max, true median gradient, native attack space, --pgd-select
best, one restart; plus --n / --iters / --eps when given), and writes
tab_adaptive_<model>.tex: one row per target, each cell the mean over seeds
with the standard deviation in parentheses when there are 2 or more seeds.
It prints every file it used and every file it skipped with the reason, and
check (i) (the share of images where the adaptive attack reaches a lower
objective than the oblivious one). No number is typed by hand. Stdlib only.
"""
import argparse
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
COLS = ("clean_defended_mAP", "oblivious_attacked_undefended_mAP",
        "oblivious_attacked_defended_mAP", "adaptive_attacked_undefended_mAP",
        "adaptive_attacked_defended_mAP")
PAPER_FLAGS = {"eot_loss": "max", "restarts": 1, "median_surrogate": "true",
               "quantize": "truncate", "attack_space": "native",
               "pgd_select": "best"}
ORDER = ("median", "jpeg", "gaussian", "tvm", "blur_tvm",
         "ens_jpeg_median_gaussian", "ens_jpeg_median_tvm",
         "ens_blur_tvm_combo", "ens_4way")


def load(model, results_dir, want):
    """{target: [payload, ...]} for the runs matching the paper's flags."""
    runs = {}
    pattern = os.path.join(results_dir, f"adaptive_bpda_{model}",
                           f"summary_adaptive_bpda_{model}_*_latest.json")
    for path in sorted(glob.glob(pattern)):
        d = json.load(open(path))
        flags, atk = d.get("attack_flags", {}), d.get("attack", {})
        why = [f"{k}={flags.get(k)}" for k, v in PAPER_FLAGS.items()
               if flags.get(k) != v]
        if d.get("smoke_test"):
            why.append("smoke test")
        for key, got in (("n", d.get("num_images")),
                         ("iters", atk.get("iters")), ("eps", atk.get("eps"))):
            if want.get(key) is not None and got != want[key]:
                why.append(f"{key}={got}")
        name = os.path.basename(path)
        if why:
            print(f"  skip {name}: {', '.join(why)}")
            continue
        print(f"  use  {name}: n={d['num_images']} iters={atk.get('iters')} "
              f"alpha={atk.get('alpha')} seed={atk.get('seed')}")
        runs.setdefault(d["target"], []).append(d)
    return runs


def cell(values):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return f"{mean:.3f}"
    return f"{mean:.3f} ({statistics.stdev(values):.3f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["yolo", "florence"], default=None)
    ap.add_argument("--results-dir",
                    default=os.path.join(ROOT, "results", "phase4"))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--eps", type=float, default=None)
    a = ap.parse_args()
    for model in ([a.model] if a.model else ["yolo", "florence"]):
        print(f"[{model}]")
        runs = load(model, a.results_dir,
                    {"n": a.n, "iters": a.iters, "eps": a.eps})
        if not runs:
            print("  no matching summaries; nothing written")
            continue
        lines = []
        for target in sorted(runs, key=lambda t: (ORDER.index(t)
                                                  if t in ORDER else 99, t)):
            rs = runs[target]
            configs = {(r["num_images"], r["attack"]["iters"],
                        r["attack"]["alpha"], r["attack"]["eps"]) for r in rs}
            assert len(configs) == 1, (
                f"{target}: runs with different n / iters / alpha / eps "
                f"{sorted(configs)}; narrow them with --n / --iters / --eps")
            label = rs[0]["defense"]["name"]
            seeds = sorted(r["attack"]["seed"] for r in rs)
            cells = [cell([r["headline"][c] for r in rs]) for c in COLS]
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
            frac = [r["surrogate_objective"]
                    ["frac_images_adaptive_below_oblivious"] for r in rs]
            print(f"  {label}: seeds {seeds}; adaptive objective below the "
                  f"oblivious one on {min(frac):.3f} to {max(frac):.3f} of "
                  f"images")
        out = os.path.join(HERE, f"tab_adaptive_{model}.tex")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("  wrote", out)


if __name__ == "__main__":
    main()
