#!/usr/bin/env python3
"""Write every results table of README.md from the committed result files.

    python3 scripts/make_readme_tables.py           # rewrite the tables in README.md
    python3 scripts/make_readme_tables.py --check   # exit 1 if README.md differs
    python3 scripts/make_readme_tables.py --print   # also print the numbers the prose cites

Each table sits in README.md between
    <!-- table:NAME -->  and  <!-- /table:NAME -->
and is replaced as a whole; text outside the markers is never touched. Every
cell is read from a summary file (or, for the two N=5,000 tables whose summary
files are not in this repo, from the paper's tex table parsed from the earlier
paper). Nothing is typed by hand. Standard library only, read-only on results/.
"""
import argparse
import decimal
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R4 = os.path.join(ROOT, "results", "phase4")
PD = os.path.join(ROOT, "docs", "paper_draft")
README = os.path.join(ROOT, "README.md")
MINUS = "\u2212"

LABEL = {"ens_jpeg_median_gaussian": "EnsJMG", "ens_jpeg_median_tvm": "EnsJMT",
         "ens_4way": "Ens4W", "ens_blur_tvm_combo": "EnsBTC",
         "ens_bilateral_median": "EnsBMT", "ens_jpeg_bilateral_tvm": "EnsJBT",
         "ens_jpeg_tvm_svd": "EnsJTS", "jpeg": "JPEG", "median": "Median",
         "gaussian": "Gaussian", "tvm": "TVM", "blur_tvm": "Blur+TVM",
         "bilateral": "Bilateral", "svd": "SVD", "bit_depth": "Bit depth",
         "dithering": "Dithering", "nlm": "NLM",
         "anisotropic": "Anisotropic diffusion", "bm3d": "BM3D",
         "random_resize": "Random resize"}
USED = set()
CITED = {}          # numbers the prose uses, printed with --print


# ---------------------------------------------------------------- helpers
def load(path):
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    USED.add(os.path.relpath(path, ROOT))
    with open(path) as f:
        return json.load(f)


def one(pattern):
    hits = sorted(glob.glob(os.path.join(ROOT, pattern)))
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


def neg(s):
    return s.replace("-", MINUS)


def rnd(x, digits, sign=False):
    """Round half up from the shortest decimal form of x (so an accuracy of exactly
    0.7535 shows as 0.754, not as the binary float's 0.753)."""
    q = decimal.Decimal(repr(float(x))).quantize(decimal.Decimal(1).scaleb(-digits),
                                                   rounding=decimal.ROUND_HALF_UP)
    s = f"{q:f}"
    if sign and not s.startswith("-"):
        s = "+" + s
    return neg(s)


def v3(x):
    return rnd(x, 3)


def v4(x):
    return rnd(x, 4)


def d3(x):
    return rnd(x, 3, sign=True)


def pct(x):
    return "0%" if rnd(x, 0) in ("0", MINUS + "0") else rnd(x, 0) + "%"


def vals(xs):
    """One value, or the range over runs that should agree but are not bit-repeatable."""
    xs = sorted({round(x, 9) for x in xs})
    if len(xs) == 1:
        return v3(xs[0])
    return f"{v3(xs[0])}–{v3(xs[-1])} ({len(xs)} runs)"


def ci(d, digits=3):
    s = (f"{rnd(d['point'], digits, True)} [{rnd(d['lo'], digits, True)}, "
         f"{rnd(d['hi'], digits, True)}]")
    return f"**{s}**" if d["excludes_0"] else s


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        assert len(r) == len(header), (header, r)
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def mean_sd(vals):
    if max(vals) - min(vals) < 1e-12:
        return v3(vals[0])
    return f"{v3(statistics.fmean(vals))} (sd {statistics.stdev(vals):.3f})"


# ---------------------------------------------------------- adaptive runs
def adaptive_runs():
    """Every adaptive summary (_latest.json) and first-300 re-score, one dict each."""
    runs = []
    for p in sorted(glob.glob(os.path.join(R4, "adaptive_bpda_*", "summary_adaptive_*_latest.json"))):
        s = load(p)
        a, f = s["attack"], s["attack_flags"]
        runs.append(dict(
            model=s["model_key"], target=s["target"], seed=a["seed"],
            obj=f["eot_loss"], surrogate=f["median_surrogate"], eps=a["eps"],
            alpha=a["alpha"], iters=a["iters"], n=s["num_images"], h=s["headline"],
            frac=s["surrogate_objective"]["frac_images_adaptive_below_oblivious"],
            branches=len(s["defense"]["branches"]), spi=s["seconds_per_image"],
            dir=os.path.basename(os.path.dirname(p)), subset=False))
    subs = glob.glob(os.path.join(R4, "*", "summary_subset_*first300.json"))
    seen = {}
    for p in sorted(subs):
        s = load(p)
        key = (s["source_summary"], s["first_k_images"])
        if key in seen:     # the same re-score written to two directories
            assert seen[key] == s["headline"], (p, key)
            continue
        seen[key] = s["headline"]
        src = load(s["source_summary"])
        a = s["attack"]
        runs.append(dict(
            model=src["model_key"], target=s["target"], seed=a["seed"],
            obj=s["attack_flags"]["eot_loss"],
            surrogate=s["attack_flags"]["median_surrogate"], eps=a["eps"],
            alpha=a["alpha"], iters=a["iters"], n=s["first_k_images"], h=s["headline"], frac=None,
            branches=len(src["defense"]["branches"]), spi=None,
            dir=os.path.basename(os.path.dirname(p)), subset=True))
    return runs


def find(runs, **kw):
    hits = [r for r in runs if all(r[k] == v for k, v in kw.items())]
    return hits


def get(runs, **kw):
    hits = find(runs, **kw)
    assert len(hits) == 1, (kw, [(h["dir"], h["subset"]) for h in hits])
    return hits[0]


def budget_name(eps):
    return {0.03: "7/255", 0.016: "4/255", 0.1: "eps 0.1"}[eps]


def model_name(m):
    return {"yolo": "YOLO-World", "florence": "Florence-2", "owl": "OWL-ViT"}[m]


# ------------------------------------------------------------- Phase 4 tables
def t_ensembles():
    c = load("results/phase4/clean_control_yolo/summary_clean_control_yolo_alltransforms_latest.json")
    rows = []
    for k, members in c["ensembles"].items():
        v2 = "yes" if k in ("ens_blur_tvm_combo", "ens_jpeg_median_gaussian",
                           "ens_jpeg_median_tvm") else "no (added in the survey)"
        rows.append([c["ensemble_paper_names"][k], f"`{k}`",
                     ", ".join(LABEL[m] for m in members), v2])
    return md(["Ensemble", "Code name", "Members (merged by class-aware NMS, IoU 0.5)",
               "In the Phase-3 v2 runs"], rows)


def adaptive_summary(runs, model, targets, seeds):
    rows = []
    base = [r for r in runs if r["model"] == model and r["eps"] == 0.03 and r["n"] == 1000
            and not r["subset"] and r["iters"] == 50 and r["surrogate"] == "true"]
    first = [r for r in base if r["obj"] == "max"]
    rows.append(["No defense", ", ".join(str(s) for s in sorted({r["seed"] for r in first})),
                 vals([r["h"]["clean_mAP"] for r in first]),
                 "", mean_sd([r["h"]["oblivious_attacked_undefended_mAP"]
                              for r in first if r["target"] == targets[0]]),
                 "", "", "", "", ""])
    for t in targets:
        mx = sorted([r for r in base if r["target"] == t and r["obj"] == "max"
                     and r["seed"] in seeds], key=lambda r: r["seed"])
        mn = [r for r in base if r["target"] == t and r["obj"] == "mean" and r["seed"] == 1]
        assert mx, (model, t)
        h = lambda k: [r["h"][k] for r in mx]                     # noqa: E731
        s1 = [r for r in mx if r["seed"] == 1][0]
        single = s1["branches"] == 1
        mean_cell = "same (one branch)" if single else (v3(mn[0]["h"]["adaptive_attacked_defended_mAP"]) if mn else "not run")
        worst = s1["h"]["adaptive_attacked_defended_mAP"]
        if mn and not single:
            worst = min(worst, mn[0]["h"]["adaptive_attacked_defended_mAP"])
        worst_cell = f"**{v3(worst)}**" + (" (max only)" if not single and not mn else "")
        rows.append([LABEL[t], ", ".join(str(r["seed"]) for r in mx),
                     mean_sd(h("clean_mAP")), mean_sd(h("clean_defended_mAP")),
                     mean_sd(h("oblivious_attacked_undefended_mAP")),
                     mean_sd(h("oblivious_attacked_defended_mAP")),
                     mean_sd(h("adaptive_attacked_defended_mAP")), mean_cell,
                     worst_cell, mean_sd(h("adaptive_attacked_undefended_mAP"))])
        CITED[f"{model}/{t}/worst_seed1"] = worst
    return md(["Defense", "Seeds", "Clean, no defense", "Clean + defense",
               "Oblivious PGD, no defense", "Oblivious PGD + defense",
               "Adaptive + defense, max objective", "Adaptive + defense, mean objective (seed 1)",
               "Worst case (seed 1)", "Adaptive image (max objective), defense removed"], rows)


def variant(r):
    v = []
    if r["eps"] != 0.03:
        v.append(budget_name(r["eps"]))
    if r["iters"] != 50:
        v.append(f"{r['iters']} steps")
    if r["surrogate"] != "true":
        v.append(f"median gradient: {r['surrogate']}")
    if r["subset"]:
        v.append("first 300 of the 1,000-image run")
    return ", ".join(v) or "main setting"


def t_all_adaptive(runs):
    order = {"yolo": 0, "florence": 1, "owl": 2}
    tgt = {"median": 0, "jpeg": 1, "gaussian": 2, "ens_jpeg_median_gaussian": 3, "ens_4way": 4}
    rs = sorted(runs, key=lambda r: (order[r["model"]], -r["eps"] if r["eps"] != 0.1 else 1,
                                     -r["n"], r["subset"], r["iters"], tgt[r["target"]],
                                     r["surrogate"] != "true", r["obj"], r["seed"]))
    rows = []
    for r in rs:
        h = r["h"]
        frac = "" if r["frac"] is None else f"{round(r['frac'] * r['n'])}/{r['n']}"
        rows.append([model_name(r["model"]), LABEL[r["target"]], r["seed"],
                     r["obj"] if r["branches"] > 1 else "(one branch)", budget_name(r["eps"]),
                     r["iters"], f"{r['n']:,}", variant(r),
                     v4(h["clean_mAP"]), v4(h["clean_defended_mAP"]),
                     v4(h["oblivious_attacked_undefended_mAP"]),
                     v4(h["oblivious_attacked_defended_mAP"]),
                     v4(h["adaptive_attacked_undefended_mAP"]),
                     f"**{v4(h['adaptive_attacked_defended_mAP'])}**", frac,
                     f"`{r['dir']}`"])
    return md(["Model", "Target", "Seed", "Objective", "Budget", "Steps", "Images", "Note",
               "Clean", "Clean + def.", "Obl., no def.", "Obl. + def.",
               "Adapt. image, def. removed", "Adapt. + def.",
               "Adaptive objective below oblivious", "Directory (`results/phase4/`)"], rows)


def t_checks(runs):
    main = [r for r in runs if not r["subset"] and r["n"] == 1000]
    fr = [round(r["frac"] * 1000) for r in main]
    fr300 = [round(r["frac"] * r["n"]) for r in runs if not r["subset"] and r["n"] == 300]
    i50 = get(runs, dir="adaptive_bpda_yolo_iters50_n300")["h"]["adaptive_attacked_defended_mAP"]
    i100 = get(runs, dir="adaptive_bpda_yolo_iters100_n300")["h"]["adaptive_attacked_defended_mAP"]
    e01 = get(runs, dir="adaptive_bpda_yolo_eps0.1")["h"]
    ad = lambda **kw: get(runs, **kw)["h"]["adaptive_attacked_defended_mAP"]   # noqa: E731
    y_jmg = (ad(model="yolo", target="ens_jpeg_median_gaussian", seed=1, obj="max", n=1000, subset=False),
             ad(model="yolo", target="ens_jpeg_median_gaussian", seed=1, obj="mean", n=1000, subset=False))
    y_4w = (ad(model="yolo", target="ens_4way", seed=1, obj="max", n=1000, subset=False),
            ad(model="yolo", target="ens_4way", seed=1, obj="mean", n=1000, subset=False))
    f_jmg = (ad(model="florence", target="ens_jpeg_median_gaussian", obj="max", n=300, subset=True, eps=0.03),
             ad(model="florence", target="ens_jpeg_median_gaussian", obj="mean", n=300, subset=False, eps=0.03))
    o_jmg = (ad(model="owl", target="ens_jpeg_median_gaussian", obj="max", n=300, subset=True, eps=0.03),
             ad(model="owl", target="ens_jpeg_median_gaussian", obj="mean", n=300, subset=False, eps=0.03))
    med = (ad(model="yolo", target="median", seed=1, surrogate="true", n=1000, subset=False),
           ad(model="yolo", target="median", seed=1, surrogate="identity", n=1000, subset=False))
    rep = {}
    for m in ("yolo", "florence", "owl"):
        vals = [r["h"]["oblivious_attacked_undefended_mAP"] for r in main
                if r["model"] == m and r["seed"] == 1 and r["eps"] == 0.03]
        if m == "owl":     # the survey runs the same oblivious PGD (seed 1, same images)
            o = load("results/phase4/oblivious_survey_owl/summary_oblivious_survey_owl_seed1_latest.json")
            assert (o["attack"]["seed"], o["attack"]["iters"], o["attack"]["alpha"]) == (1, 50, 0.002)
            vals.append(o["headline"]["attacked_mAP"])
        rep[m] = (min(vals), max(vals), len(vals))
    rows = [
        ["Adaptive vs oblivious objective", f"all {len(main)} runs on 1,000 images (3 models)",
         f"the adaptive objective is below the oblivious one on {min(fr):,}–{max(fr):,} of 1,000 images, "
         f"and on {'all 300' if min(fr300) == 300 else f'{min(fr300)}–{max(fr300)}'} images of every 300-image run",
         "the optimizer does what it should"],
        ["More steps", "YOLO-World EnsJMG, 300 images",
         f"50 steps {v3(i50)} → 100 steps {v3(i100)}",
         "not converged at 50 steps: the adaptive numbers are **upper bounds**"],
        ["Larger budget", f"YOLO-World EnsJMG, eps 0.1 (alpha {get(runs, dir='adaptive_bpda_yolo_eps0.1')['alpha']}), 300 images",
         f"adaptive + defense {v3(e01['adaptive_attacked_defended_mAP'])} (oblivious + defense {v3(e01['oblivious_attacked_defended_mAP'])})",
         "mAP keeps falling as eps grows: no sign of gradient masking"],
        ["Max vs mean objective", "YOLO-World EnsJMG and Ens4W (1,000 images); Florence-2 and OWL-ViT EnsJMG (first 300)",
         f"YOLO-World: mean is stronger ({v3(y_jmg[0])} → {v3(y_jmg[1])}, {v3(y_4w[0])} → {v3(y_4w[1])}). "
         f"OWL-ViT: mean is stronger ({v3(o_jmg[0])} → {v3(o_jmg[1])}). "
         f"Florence-2: within {v3(abs(f_jmg[1] - f_jmg[0]))} ({v3(f_jmg[0])} max vs {v3(f_jmg[1])} mean)",
         "report the **worst case over objectives**"],
        ["Median gradient", "YOLO-World Median, true vs identity (BPDA), 1,000 images",
         f"{v3(med[0])} vs {v3(med[1])}", "the true median gradient is the stronger attack"],
        ["Repeatability", "same seed and images, `--deterministic`: oblivious PGD with no defense, "
         "across the runs that compute it",
         "; ".join(f"{model_name(m)} " + (f"{v4(lo)} in all {n} runs (identical)" if lo == hi
                                           else f"{v4(lo)}–{v4(hi)} over {n} runs")
                   for m, (lo, hi, n) in rep.items()),
         "YOLO-World is bit-repeatable; Florence-2 and OWL-ViT are not"],
    ]
    CITED.update({"checks/frac_range": (min(fr), max(fr)), "checks/rep": rep})
    return md(["Check", "Setting", "Result", "Meaning"], rows)


def t_clean_cost():
    y = load("results/phase4/clean_control_yolo/summary_clean_control_yolo_alltransforms_latest.json")["tracks"]["yolo"]
    fd = load("results/phase4/clean_control_florence_det/summary_clean_control_florence_alltransforms_latest.json")["tracks"]["florence_det"]
    fo = load("results/phase4/clean_control_florence_ocr/summary_clean_control_florence_alltransforms_latest.json")["tracks"]["florence_ocr"]
    for t in (y, fd, fo):
        assert t["num_images"] == 5000 and not t["failures"]
    rows = [["**No defense**", v4(y["undefended_clean"]), "", v4(fd["undefended_clean"]), "",
             v3(fo["undefended_clean"]), ""]]
    keys = list(dict.fromkeys(list(y["defenses"]) + list(fo["defenses"])))
    ens = sorted([k for k in keys if k.startswith("ens_")], key=lambda k: -y["defenses"][k]["delta_vs_measured"])
    solo = sorted([k for k in keys if not k.startswith("ens_")],
                  key=lambda k: -(y["defenses"][k]["delta_vs_measured"] if k in y["defenses"] else -9))
    for k in ens + solo:
        cells = [LABEL[k]]
        for t, f in ((y, v4), (fd, v4), (fo, v3)):
            d = t["defenses"].get(k)
            cells += ["", ""] if d is None else [f(d["value"]), d3(d["delta_vs_measured"])]
        rows.append(cells)
    return md(["Defense", "YOLO-World mAP", "change", "Florence-2 det. mAP", "change",
               "OCR self-consistency", "change"], rows)


def t_smolvlm():
    s = load("results/phase4/presence_qa_smolvlm/summary_presence_qa_smolvlm_latest.json")
    h = s["headline"]
    rows = [["**No defense**", v3(h["clean_accuracy"]), "", v3(h["attacked_accuracy"]), "", ""]]
    for k in sorted(s["defenses"], key=lambda k: -s["defenses"][k]["attacked_accuracy"]):
        d = s["defenses"][k]
        rows.append([LABEL[k], v3(d["clean_accuracy"]), ci(d["clean_cost"]),
                     v3(d["attacked_accuracy"]), ci(d["recovery"]), pct(d["recovery_pct_of_damage"])])
    CITED["smolvlm"] = (s["num_images"], s["num_questions"], s["share_yes_labels"], h)
    return md(["Defense", "Clean accuracy", "Clean cost [95% CI]", "Attacked accuracy",
               "Recovery [95% CI]", "Share of damage recovered"], rows)


def t_owl_survey():
    o = load("results/phase4/oblivious_survey_owl/summary_oblivious_survey_owl_seed1_latest.json")
    b = load("results/phase4/oblivious_survey_owl/summary_bootstrap_oblivious_survey_owl_seed1.json")["ensembles"]
    h = o["headline"]
    rows = [["**No defense**", v3(h["clean_mAP"]), "", v3(h["attacked_mAP"]), "", "", ""]]
    for k in sorted(o["defenses"], key=lambda k: -o["defenses"][k]["attacked_mAP"]):
        d = o["defenses"][k]
        vs, cc = "", d3(d["clean_cost"])
        if k in b:
            vs = f"{ci(b[k]['ensemble_minus_best_member_attacked'])} ({LABEL[b[k]['best_member_under_attack']]})"
            cc = ci(b[k]["clean_cost"])
            assert abs(b[k]["clean_cost"]["point"] - d["clean_cost"]) < 1e-9
        rows.append([LABEL[k], v3(d["clean_mAP"]), cc, v3(d["attacked_mAP"]), d3(d["recovery"]),
                     pct(d["recovery_pct_of_damage"]), vs])
    CITED["owl_survey"] = h
    return md(["Defense", "Clean + defense", "Clean cost (ensembles: [95% CI])",
               "Attacked + defense", "Recovery", "Share of damage recovered",
               "Ensemble − best member, attacked [95% CI] (member)"], rows)


def t_owl_adaptive(runs):
    rows = []
    for r in sorted(find(runs, model="owl"), key=lambda r: (-r["n"], r["subset"], r["target"] != "median", r["obj"])):
        h = r["h"]
        rows.append([LABEL[r["target"]], r["obj"] if r["branches"] > 1 else "(one branch)",
                     f"{r['n']:,}" + (" (first 300 of the 1,000-image run)" if r["subset"] else ""),
                     v3(h["clean_mAP"]), v3(h["clean_defended_mAP"]),
                     v3(h["oblivious_attacked_undefended_mAP"]), v3(h["oblivious_attacked_defended_mAP"]),
                     f"**{v3(h['adaptive_attacked_defended_mAP'])}**",
                     v3(h["adaptive_attacked_undefended_mAP"])])
    return md(["Target", "Objective", "Images", "Clean, no defense", "Clean + defense",
               "Oblivious PGD, no defense", "Oblivious PGD + defense",
               "**Adaptive + defense**", "Adaptive image, defense removed"], rows)


def t_budget(runs):
    rows = []
    for model, targets in (("yolo", ("median", "jpeg", "gaussian", "ens_jpeg_median_gaussian")),
                           ("florence", ("median", "ens_jpeg_median_gaussian"))):
        b = [r for r in runs if r["model"] == model and r["n"] == 300 and r["iters"] == 50
             and r["eps"] in (0.016, 0.03) and r["surrogate"] == "true"]
        und = {e: [r["h"]["oblivious_attacked_undefended_mAP"] for r in b if r["eps"] == e]
               for e in (0.016, 0.03)}
        cl = [r["h"]["clean_mAP"] for r in b]
        rows.append([model_name(model), "no defense", "", vals(cl), vals(und[0.016]), "(same)",
                     vals(und[0.03]), "(same)"])
        for t in targets:
            objs = ("max", "mean") if t.startswith("ens_") else ("max",)
            for obj in objs:
                cells = {}
                for e in (0.016, 0.03):
                    hit = [r for r in b if r["target"] == t and r["eps"] == e and r["obj"] == obj]
                    # YOLO EnsJMG 7/255: a separate 300-image run and the first-300
                    # re-score of the 1,000-image run; YOLO is bit-repeatable, so equal.
                    for x in hit[1:]:
                        for k in ("clean_defended_mAP", "oblivious_attacked_defended_mAP",
                                  "adaptive_attacked_defended_mAP"):
                            assert abs(x["h"][k] - hit[0]["h"][k]) < 1e-9, (model, t, e, obj, k)
                    cells[e] = hit[0] if hit else None
                cdef = [c["h"]["clean_defended_mAP"] for c in cells.values() if c]
                row = ["", LABEL[t], obj if t.startswith("ens_") else "(one branch)", vals(cdef)]
                for e in (0.016, 0.03):
                    c = cells[e]
                    row += (["not run", "not run"] if c is None else
                            [v3(c["h"]["oblivious_attacked_defended_mAP"]),
                             f"**{v3(c['h']['adaptive_attacked_defended_mAP'])}**"])
                rows.append(row)
    return md(["Model", "Defense", "Objective", "Clean + defense",
               "4/255 oblivious + defense", "**4/255 adaptive + defense**",
               "7/255 oblivious + defense", "**7/255 adaptive + defense**"], rows)


def v2_track(name):
    return load(f"results/{name}/summary.json")


V2 = [("YOLO-World · FGSM", "results_phase3_yolo_fgsm_v2", "FGSM_Phase3_YOLO_v2.ipynb"),
      ("YOLO-World · PGD (10 steps)", "results_phase3_yolo_pgd_v2", "PGD_Phase3_YOLO_v2.ipynb"),
      ("YOLO-World · Patch", "results_phase3_yolo_patch_v2", "Patch_Phase3_YOLO_v2.ipynb"),
      ("Florence-2 det. · FGSM", "results_phase3_florence_fgsm_v2", "FGSM_Phase3_Florence_v2.py"),
      ("Florence-2 det. · PGD", "results_phase3_florence_pgd_v2", "PGD_Phase3_Florence_v2.py"),
      ("Florence-2 det. · Patch", "results_phase3_florence_patch_v2", "Patch_Phase3_Florence_v2.py"),
      ("Florence-2 OCR · FGSM", "results_fgsm_florence2_ocr_robust", "FGSM_Florence2_OCR_Robust.py"),
      ("Florence-2 OCR · PGD (alpha 0.003)", "results_pgd_florence2_ocr_robust", "PGD_Florence2_OCR_Robust.py"),
      ("Florence-2 OCR · Patch", "results_patch_florence2_ocr_robust", "Patch_Florence2_OCR_Robust.py")]
V2_ORDER = ["jpeg", "median", "tvm", "gaussian", "blur_tvm", "ens_blur_tvm_combo",
            "ens_jpeg_median_gaussian", "ens_jpeg_median_tvm"]


def v2_values(d):
    """(clean, attacked, {defense: (defended, recovery_pct)}) for either v2 summary layout."""
    if "defenses" in d:
        assert d["num_images"] == 1000
        return d["clean_mAP"], d["attacked_mAP"], {k: (v["mAP"], v["recovery_pct"]) for k, v in d["defenses"].items()}
    assert d["images"] == 1000 and d["failed_images"] == 0
    return d["clean_baseline"], d["attacked"], {r["name"]: (r["recovered"], r["recovery_pct"]) for r in d["ranked_defenses"]}


def t_v2_best():
    rows = []
    for name, dname, script in V2:
        c, a, defs = v2_values(v2_track(dname))
        best = max(defs, key=lambda k: defs[k][0])
        rows.append([name, v4(c), v4(a), LABEL[best], f"**{v4(defs[best][0])}**",
                     rnd(defs[best][1], 1, True) + "%", f"`{script}`", f"`results/{dname}/summary.json`"])
    return md(["Track", "Clean", "Attacked", "Best defense", "Defended", "Recovery",
               "Generated by", "Source file"], rows)


def t_v2_full():
    cols = [v2_values(v2_track(d)) for _, d, _ in V2]
    for _, _, defs in cols:
        assert sorted(defs) == sorted(V2_ORDER)
    best = [max(defs, key=lambda k: defs[k][0]) for _, _, defs in cols]
    rows = [["Clean, no defense"] + [v4(c) for c, _, _ in cols],
            ["Attacked, no defense"] + [v4(a) for _, a, _ in cols]]
    for k in V2_ORDER:
        r = [LABEL[k]]
        for (_, _, defs), b in zip(cols, best):
            s = f"{v4(defs[k][0])} ({rnd(defs[k][1], 0, True)}%)"
            r.append(f"**{s}**" if k == b else s)
        rows.append(r)
    return md([""] + [n.replace(" (10 steps)", "").replace(" (alpha 0.003)", "") for n, _, _ in V2], rows)


def t_ci(model):
    d = load("results/phase4/rescore_phase3_v2/summary_rescore_phase3_v2_latest.json")
    atk = ("fgsm", "pgd", "patch")
    rec = {a: {r["defense"]: r for r in d["recovery"][model][a]["defenses"]} for a in atk}
    clean = d["clean_point"][model]["mAP"]
    rows = [["**No defense**", v3(clean), ""] +
            sum([[v3(d["recovery"][model][a]["attacked_mAP"]), ""] for a in atk], [])]
    for r in d["clean_cost"][model]:
        k = r["defense"]
        row = [r["label"], v3(r["mAP"]), ci(r["delta"]["mAP"])]
        for a in atk:
            row += [v3(rec[a][k]["mAP"]), ci(rec[a][k]["delta"]["mAP"])]
        rows.append(row)
    return md(["Defense", "Clean + defense", "Clean cost [95% CI]", "FGSM + defense",
               "Recovery [95% CI]", "PGD + defense", "Recovery [95% CI]",
               "Patch + defense", "Recovery [95% CI]"], rows)


def florence5k():
    out = {}
    for a in ("fgsm", "pgd", "patch"):
        f = load(f"results/results_survey_florence_detection_{a}/summary_{a}.json")
        assert int(f["num_images"]) == 5000 and not f["failures"]
        out[a] = f
    return out


def t_survey_best():
    f = florence5k()
    rows = []
    for a, name in (("fgsm", "FGSM"), ("pgd", "PGD"), ("patch", "Patch")):
        s = f[a]
        best = max(s["defenses"], key=lambda k: s["defenses"][k]["mAP"])
        solo = max((k for k in s["defenses"] if not k.startswith("ens_")), key=lambda k: s["defenses"][k]["mAP"])
        d = s["defenses"][best]
        rows.append([f"Florence-2 det. · {name}", v4(s["clean_mAP"]), v4(s["attacked_mAP"]),
                     LABEL[best], f"**{v4(d['mAP'])}**", rnd(d["recovery_pct"], 1, True) + "%",
                     f"{LABEL[solo]} {v4(s['defenses'][solo]['mAP'])}",
                     f"`results/results_survey_florence_detection_{a}/summary_{a}.json`"])
    return md(["Track", "Clean", "Attacked", "Best defense", "Defended", "Recovery",
               "Best single transform", "Source file"], rows)


def t_florence5k_full():
    f = florence5k()
    keys = list(f["pgd"]["defenses"])
    solo = sorted([k for k in keys if not k.startswith("ens_")], key=lambda k: -f["pgd"]["defenses"][k]["mAP"])
    ens = sorted([k for k in keys if k.startswith("ens_")], key=lambda k: -f["pgd"]["defenses"][k]["mAP"])
    rows = [["No attack, no defense"] + sum([[v4(f[a]["clean_mAP"]), ""] for a in ("fgsm", "pgd", "patch")], []),
            ["Attacked, no defense"] + sum([[v4(f[a]["attacked_mAP"]), ""] for a in ("fgsm", "pgd", "patch")], []),
            ["*Single transforms*"] + [""] * 6]
    for group in (solo, ["*Ensembles*"], ens):
        for k in group:
            if k.startswith("*"):
                rows.append([k] + [""] * 6)
                continue
            r = [LABEL[k] + (" †" if k == "random_resize" else "")]
            for a in ("fgsm", "pgd", "patch"):
                d = f[a]["defenses"][k]
                r += [v4(d["mAP"]), rnd(d["recovery_pct"], 1)]
            rows.append(r)
    return md(["", "FGSM mAP", "Rec. %", "PGD mAP", "Rec. %", "Patch mAP", "Rec. %"], rows)


def tex_rows(name):
    out = []
    for line in open(os.path.join(PD, name)):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith("\\midrule"):
            m = re.search(r"\\emph\{([^}]*)\}", line)
            if m:
                out.append([f"*{m.group(1)}*"])
            continue
        line = re.sub(r"\\\\$", "", line).strip()
        line = re.sub(r"\$(-[\d.]+)\$", lambda m: m.group(1), line)
        for a, b in (("($\\sigma{=}1$)", "(σ=1)"), ("($w{=}0.05$)", "(w=0.05)"),
                     ("($3{\\times}3$)", "(3×3)"), ("($q{=}75$)", "(q=75)"),
                     ("($h{=}0.8\\hat\\sigma$)", "(h=0.8σ̂)"), ("90\\%", "90%")):
            line = line.replace(a, b)
        assert "$" not in line and "\\" not in line, (name, line)
        out.append([neg(c.strip()) if re.fullmatch(r"-[\d.]+", c.strip())
                    else ("" if c.strip() == "--" else c.strip()) for c in line.split("&")])
    return out


def t_tex(name, header):
    rows = []
    for r in tex_rows(name):
        rows.append(r + [""] * (len(header) - len(r)) if len(r) == 1 else r)
    return md(header, rows)


# ------------------------------------------------------------- Phase 3 vs 4
def t_p3_vs_p4(runs):
    v2 = v2_values(v2_track("results_phase3_yolo_pgd_v2"))
    main = [r for r in runs if r["model"] == "yolo" and r["n"] == 1000 and not r["subset"]
            and r["eps"] == 0.03 and r["iters"] == 50 and r["surrogate"] == "true"]
    obl = lambda t: statistics.fmean(r["h"]["oblivious_attacked_defended_mAP"] for r in main  # noqa: E731
                                     if r["target"] == t and r["obj"] == "max")
    und = statistics.fmean(r["h"]["oblivious_attacked_undefended_mAP"] for r in main
                           if r["target"] == "median" and r["obj"] == "max")
    rows = [["Attack", "PGD, 10 steps, alpha 0.0075, letterbox space",
             "PGD, 50 steps, alpha 0.002, native resolution",
             "BPDA+EOT through the defense, 50 steps"],
            ["Clean, no defense", v4(v2[0]), v3(main[0]["h"]["clean_mAP"]), v3(main[0]["h"]["clean_mAP"])],
            ["Attacked, no defense", v4(v2[1]), f"{v3(und)} (3 seeds)", "(no defense to adapt to)"]]
    for t in ("median", "ens_jpeg_median_gaussian", "ens_4way", "ens_blur_tvm_combo"):
        p3 = v4(v2[2][t][0]) if t in v2[2] else "not in v2"
        have = [r for r in main if r["target"] == t]
        p4o = f"{v3(obl(t))}" + (" (3 seeds)" if len({r['seed'] for r in have}) == 3 else " (seed 1)") if have else "not run"
        p4a = v3(CITED[f"yolo/{t}/worst_seed1"]) + " (seed 1)" if have else "not run"
        rows.append([LABEL[t], p3, p4o, p4a])
    return md(["YOLO-World, same 1,000 images", "Phase 3 (v2)", "Phase 4, oblivious",
               "Phase 4, adaptive (worst case)"], rows)


# ------------------------------------------------------------------- driver
def tables():
    runs = adaptive_runs()
    t = {
        "ensembles": t_ensembles(),
        "p4_yolo_adaptive": adaptive_summary(runs, "yolo", ["median", "ens_jpeg_median_gaussian", "ens_4way"], (1, 2, 3)),
        "p4_florence_adaptive": adaptive_summary(runs, "florence", ["median", "ens_jpeg_median_gaussian"], (1,)),
        "p4_checks": t_checks(runs),
        "p4_clean_cost": t_clean_cost(),
        "p4_smolvlm": t_smolvlm(),
        "p4_owl_survey": t_owl_survey(),
        "p4_owl_adaptive": t_owl_adaptive(runs),
        "p4_budget": t_budget(runs),
        "p4_all_adaptive": t_all_adaptive(runs),
        "p3_v2_best": t_v2_best(),
        "p3_v2_full": t_v2_full(),
        "p3_ci_yolo": t_ci("yolo"),
        "p3_ci_florence": t_ci("florence"),
        "p3_survey_best": t_survey_best(),
        "p3_n5000_florence": t_florence5k_full(),
        "p3_n5000_yolo": t_tex("tab_yolo_n5000.tex", ["", "FGSM mAP", "Rec. %", "PGD mAP", "Rec. %", "Patch mAP", "Rec. %"]),
        "p3_n5000_ocr": t_tex("tab_ocr_n5000.tex", ["", "FGSM", "PGD", "Patch"]),
    }
    t["p4_vs_p3"] = t_p3_vs_p4(runs)
    USED.add("docs/paper_draft/tab_yolo_n5000.tex")
    USED.add("docs/paper_draft/tab_ocr_n5000.tex")
    return t, runs


def apply(text, tabs):
    for name, body in tabs.items():
        pat = re.compile(rf"(<!-- table:{re.escape(name)} -->\n).*?(<!-- /table:{re.escape(name)} -->)", re.S)
        assert len(pat.findall(text)) == 1, f"README needs exactly one table:{name} marker pair"
        text = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text, count=1)
    unknown = set(re.findall(r"<!-- table:([\w]+) -->", text)) - set(tabs)
    assert not unknown, f"README has markers with no generator: {unknown}"
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    tabs, runs = tables()
    old = open(README).read()
    new = apply(old, tabs)
    if a.print:
        for k, v in CITED.items():
            print(f"  {k}: {v}")
        print("[files used]")
        for u in sorted(USED):
            print("  ", u)
    if a.check:
        if new != old:
            print("README.md tables differ from the result files; run without --check")
            sys.exit(1)
        print(f"README.md tables match the result files ({len(tabs)} tables, {len(USED)} files)")
        return
    if new != old:
        open(README, "w").write(new)
    print(f"wrote {len(tabs)} tables into README.md from {len(USED)} files")


if __name__ == "__main__":
    main()
