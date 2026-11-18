#!/usr/bin/env python3
"""Build the Phase-4 table bodies of the draft from the committed summaries.

    python docs/paper_draft/make_phase4_tables.py

Writes, next to this file (each a table body pulled in by \\tabinput):
  tab_adaptive_yolo.tex      Table 5: YOLO-World, 1,000 images, 50 steps. Five
                             columns with the max objective (mean over seeds,
                             sd in parentheses) plus the mean-objective
                             adaptive-defended cell (seed 1).
  tab_adaptive_florence.tex  Florence-2 detection, same columns (seed 1).
  tab_clean_n5000.tex        clean cost of every transform, 5,000 images:
                             YOLO-World and Florence-2 mAP change, OCR
                             self-consistency.
  tab_smolvlm.tex            SmolVLM-256M: clean accuracy, clean cost [CI],
                             attacked accuracy, recovery [CI].
  tab_owl_survey.tex         OWL-ViT, every transform under oblivious PGD
                             (written only when that summary exists).
  tab_budget.tex             defended mAP against the budget, 300 images
                             (written only when the 4/255 runs exist).
and prints every number the prose cites (checks i-v, worst cases), with the
file it came from. No number is typed by hand. Stdlib only.
"""
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(ROOT, "results", "phase4")
COLS = ("clean_defended_mAP", "oblivious_attacked_undefended_mAP",
        "oblivious_attacked_defended_mAP", "adaptive_attacked_undefended_mAP",
        "adaptive_attacked_defended_mAP")
LABEL = {"ens_jpeg_median_gaussian": "EnsJMG", "ens_jpeg_median_tvm": "EnsJMT",
         "ens_4way": "Ens4W", "ens_blur_tvm_combo": "EnsBTC",
         "ens_bilateral_median": "EnsBMT", "ens_jpeg_bilateral_tvm": "EnsJBT",
         "ens_jpeg_tvm_svd": "EnsJTS", "jpeg": "JPEG", "median": "Median",
         "gaussian": "Gaussian", "tvm": "TVM", "blur_tvm": "Blur+TVM",
         "bilateral": "Bilateral", "svd": "SVD", "bit_depth": "Bit depth",
         "dithering": "Dithering", "nlm": "NLM",
         "anisotropic": "Anisotropic diff.", "bm3d": "BM3D",
         "random_resize": "Random resize"}
ENS = [k for k in LABEL if k.startswith("ens_")]
SOLO = [k for k in LABEL if not k.startswith("ens_")]
USED = []



def rnd(x, digits, sign=False):
    """Round half up from the shortest decimal form of x (0.7535 -> 0.754)."""
    import decimal
    q = decimal.Decimal(repr(float(x))).quantize(decimal.Decimal(1).scaleb(-digits),
                                                   rounding=decimal.ROUND_HALF_UP)
    s = f"{q:f}"
    return "+" + s if sign and not s.startswith("-") else s

def load(path):
    USED.append(os.path.relpath(path, ROOT))
    with open(path) as f:
        return json.load(f)


def adaptive(model, target, seed, mean_obj=False, sub="adaptive_bpda_{m}",
             extra=""):
    flags = "_eot-loss-mean" if mean_obj else ""
    native = "" if model != "yolo" else "_attack-space-native"
    p = os.path.join(R, sub.format(m=model),
                     f"summary_adaptive_bpda_{model}_{target}_seed{seed}"
                     f"{flags}{native}_pgd-select-best{extra}_latest.json")
    return load(p) if os.path.exists(p) else None


def f3(x):
    return f"{rnd(x, 3)}"


def write(name, lines):
    with open(os.path.join(HERE, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {name} ({len(lines)} rows)")


def adaptive_table(model, targets, seeds):
    lines, check = [], {}
    for t in targets:
        runs = [r for r in (adaptive(model, t, s) for s in seeds) if r]
        assert runs, f"no {model} {t} runs"
        for r in runs:
            assert r["attack"]["iters"] == 50 and r["num_images"] == 1000
        cells = []
        for c in COLS:
            v = [r["headline"][c] for r in runs]
            cells.append(f3(statistics.fmean(v)) if len(v) < 2 else
                         f"{rnd(statistics.fmean(v), 3)} ({rnd(statistics.stdev(v), 3)})")
        mean_run = adaptive(model, t, 1, mean_obj=True)
        if len(runs[0]["defense"]["branches"]) == 1:
            cells.append("same")
        else:
            cells.append(f3(mean_run["headline"]["adaptive_attacked_defended_mAP"])
                         if mean_run else "--")
        lines.append(f"{LABEL[t]} & " + " & ".join(cells) + " \\\\")
        s1 = runs[0]["headline"]
        worst = s1["adaptive_attacked_defended_mAP"]
        if mean_run:
            worst = min(worst, mean_run["headline"]["adaptive_attacked_defended_mAP"])
        check[t] = {
            "seeds": [r["attack"]["seed"] for r in runs],
            "frac_adaptive_below_oblivious": [
                r["surrogate_objective"]["frac_images_adaptive_below_oblivious"]
                for r in runs],
            "seed1_adaptive_defended_max": s1["adaptive_attacked_defended_mAP"],
            "seed1_adaptive_defended_mean": (mean_run["headline"]
                                             ["adaptive_attacked_defended_mAP"]
                                             if mean_run else None),
            "seed1_worst_case_defended": worst,
            "seed1_oblivious_undefended": s1["oblivious_attacked_undefended_mAP"],
            "per_seed_adaptive_defended": [r["headline"]
                                           ["adaptive_attacked_defended_mAP"]
                                           for r in runs],
            "clean": s1["clean_mAP"]}
    return lines, check


def main():
    print("[adaptive]")
    lines, cy = adaptive_table("yolo", ["median", "ens_jpeg_median_gaussian",
                                        "ens_4way"], (1, 2, 3))
    write("tab_adaptive_yolo.tex", lines)
    lines, cf = adaptive_table("florence", ["median",
                                            "ens_jpeg_median_gaussian"], (1,))
    write("tab_adaptive_florence.tex", lines)
    print(json.dumps({"yolo": cy, "florence": cf}, indent=1))

    print("[checks]")
    p = os.path.join(R, "adaptive_bpda_yolo", "summary_adaptive_bpda_yolo_median"
                     "_seed1_median-surrogate-identity_attack-space-native_"
                     "pgd-select-best_latest.json")
    ident = load(p)["headline"]["adaptive_attacked_defended_mAP"]
    i50 = load(glob.glob(os.path.join(R, "adaptive_bpda_yolo_iters50_n300",
                                      "summary_*_latest.json"))[0])["headline"]
    i100 = load(glob.glob(os.path.join(R, "adaptive_bpda_yolo_iters100_n300",
                                       "summary_*_latest.json"))[0])["headline"]
    e01 = load(glob.glob(os.path.join(R, "adaptive_bpda_yolo_eps0.1",
                                      "summary_*_latest.json"))[0])
    fl_mean = load(glob.glob(os.path.join(R, "adaptive_bpda_florence_n300",
                                          "summary_adaptive*_latest.json"))[0])
    fl_max300 = load(os.path.join(R, "adaptive_bpda_florence_n300",
                                  "summary_subset_ens_jpeg_median_gaussian_"
                                  "seed1_max_first300.json"))
    print(json.dumps({
        "ii_ensjmg_300_50steps": i50["adaptive_attacked_defended_mAP"],
        "ii_ensjmg_300_100steps": i100["adaptive_attacked_defended_mAP"],
        "iii_eps0.1_alpha": e01["attack"]["alpha"],
        "iii_eps0.1_defended": e01["headline"]["adaptive_attacked_defended_mAP"],
        "iii_eps0.1_oblivious_defended": e01["headline"]
        ["oblivious_attacked_defended_mAP"],
        "v_median_identity": ident,
        "iv_florence_ensjmg_300_mean": fl_mean["headline"]
        ["adaptive_attacked_defended_mAP"],
        "iv_florence_ensjmg_300_max": fl_max300["headline"]
        ["adaptive_attacked_defended_mAP"]}, indent=1))

    print("[clean cost, 5,000 images]")
    y = load(os.path.join(R, "clean_control_yolo", "summary_clean_control_yolo_"
                          "alltransforms_latest.json"))["tracks"]["yolo"]
    fd = load(os.path.join(R, "clean_control_florence_det", "summary_clean_"
                           "control_florence_alltransforms_latest.json")
              )["tracks"]["florence_det"]
    fo = load(os.path.join(R, "clean_control_florence_ocr", "summary_clean_"
                           "control_florence_alltransforms_latest.json")
              )["tracks"]["florence_ocr"]
    print(f"  undefended clean: yolo {rnd(y['undefended_clean'], 4)}, florence "
          f"{rnd(fd['undefended_clean'], 4)}, ocr {rnd(fo['undefended_clean'], 3)}")
    lines = []
    for k in SOLO + ENS:
        cells = []
        for tr, key in ((y, "delta_vs_measured"), (fd, "delta_vs_measured"),
                        (fo, "value")):
            d = tr["defenses"].get(k)
            cells.append("" if d is None else
                         (f"{rnd(d[key], 3, True)}" if key.startswith("delta")
                          else f"{rnd(d[key], 3)}"))
        if k == "ens_jpeg_median_gaussian":
            lines.append("\\midrule")
        lines.append(f"{LABEL[k]} & " + " & ".join(cells) + " \\\\")
    lines = [l for l in lines if l == "\\midrule" or l.split(" & ", 1)[1].strip(" &\\")]
    write("tab_clean_n5000.tex", lines)

    print("[SmolVLM]")
    s = load(os.path.join(R, "presence_qa_smolvlm",
                          "summary_presence_qa_smolvlm_latest.json"))
    print("  headline", s["headline"], "questions", s["num_questions"],
          "share yes", s["share_yes_labels"])
    lines = []
    order = sorted(s["defenses"], key=lambda k: -s["defenses"][k]["attacked_accuracy"])
    for k in order:
        d = s["defenses"][k]
        cc, rc = d["clean_cost"], d["recovery"]
        b = lambda iv: "\\textbf{" if iv["excludes_0"] else "{"   # noqa: E731
        lines.append(
            f"{LABEL[k]} & {rnd(d['clean_accuracy'], 3)} & "
            f"{b(cc)}{rnd(cc['point'], 3, True)}}} [{rnd(cc['lo'], 3, True)}, {rnd(cc['hi'], 3, True)}] & "
            f"{rnd(d['attacked_accuracy'], 3)} & "
            f"{b(rc)}{rnd(rc['point'], 3, True)}}} [{rnd(rc['lo'], 3, True)}, {rnd(rc['hi'], 3, True)}] \\\\")
    write("tab_smolvlm.tex", lines)

    owl = os.path.join(R, "oblivious_survey_owl",
                       "summary_oblivious_survey_owl_seed1_latest.json")
    if os.path.exists(owl):
        print("[OWL-ViT survey]")
        o = load(owl)
        print("  headline", o["headline"], "n", o["num_images"])
        bp = os.path.join(R, "oblivious_survey_owl",
                          "summary_bootstrap_oblivious_survey_owl_seed1.json")
        boot = load(bp)["ensembles"] if os.path.exists(bp) else {}
        lines = []
        for k in sorted(o["defenses"], key=lambda k: -o["defenses"][k]["attacked_mAP"]):
            d = o["defenses"][k]
            vs = ""
            if k in boot:
                b = boot[k]["ensemble_minus_best_member_attacked"]
                bold = "\\textbf{" if b["excludes_0"] else "{"
                vs = (f"{bold}{rnd(b['point'], 3, True)}}} [{rnd(b['lo'], 3, True)}, {rnd(b['hi'], 3, True)}] "
                      f"({LABEL[boot[k]['best_member_under_attack']]})")
            lines.append(f"{LABEL[k]} & {rnd(d['clean_cost'], 3, True)} & "
                         f"{rnd(d['attacked_mAP'], 3)} & {rnd(d['recovery_pct_of_damage'], 0)}\\% & "
                         f"{vs} \\\\")
        write("tab_owl_survey.tex", lines)
        for k, r in boot.items():
            print(f"  boot {k}: vs best {rnd(r['ensemble_minus_best_member_attacked']['point'], 4, True)} "
                  f"[{rnd(r['ensemble_minus_best_member_attacked']['lo'], 4, True)}, "
                  f"{rnd(r['ensemble_minus_best_member_attacked']['hi'], 4, True)}] bc_excl "
                  f"{r['ensemble_minus_best_member_attacked']['bc_excludes_0']}; vs median "
                  f"{rnd(r['ensemble_minus_median_attacked']['point'], 4, True)} "
                  f"[{rnd(r['ensemble_minus_median_attacked']['lo'], 4, True)}, "
                  f"{rnd(r['ensemble_minus_median_attacked']['hi'], 4, True)}]; clean "
                  f"{rnd(r['clean_cost']['point'], 4, True)} [{rnd(r['clean_cost']['lo'], 4, True)}, "
                  f"{rnd(r['clean_cost']['hi'], 4, True)}]")
        for t in ("median", "ens_jpeg_median_gaussian"):
            r = adaptive("owl", t, 1)
            if r:
                print(f"  OWL adaptive {t}: n={r['num_images']}",
                      {k: round(v, 4) for k, v in r["headline"].items()},
                      "frac below", r["surrogate_objective"]
                      ["frac_images_adaptive_below_oblivious"])
        r = adaptive("owl", "ens_jpeg_median_gaussian", 1, mean_obj=True,
                     sub="adaptive_bpda_{m}_n300")
        if r:
            print("  OWL adaptive EnsJMG mean (300):",
                  {k: round(v, 4) for k, v in r["headline"].items()})

    print("[budget, 300 images]")
    rows = []
    for model, native in (("yolo", "_attack-space-native"), ("florence", "")):
        for e in (4, 7):
            d = os.path.join(R, f"adaptive_bpda_{model}_eps{e}of255_n300")
            for p in sorted(glob.glob(os.path.join(d, "summary_adaptive*_latest.json"))):
                r = load(p)
                rows.append((model, e, r["target"], r["attack_flags"]["eot_loss"],
                             r["headline"]))
        sub = os.path.join(R, "budget_curve_eps7of255_first300")
        for p in sorted(glob.glob(os.path.join(sub, f"summary_subset_{model}_*.json"))):
            r = load(p)
            rows.append((model, 7, r["target"], r["attack_flags"]["eot_loss"],
                         r["headline"]))
    for row in rows:
        h = row[4]
        print(f"  {row[0]:8} {row[1]}/255 {row[2]:26} {row[3]:4} clean "
              f"{rnd(h['clean_mAP'], 4)} obl-undef "
              f"{rnd(h['oblivious_attacked_undefended_mAP'], 4)} obl-def "
              f"{rnd(h['oblivious_attacked_defended_mAP'], 4)} adp-undef "
              f"{rnd(h['adaptive_attacked_undefended_mAP'], 4)} adp-def "
              f"{rnd(h['adaptive_attacked_defended_mAP'], 4)}")
    budget_table(rows)
    print("[files used]")
    for u in USED:
        print("  ", u)


def budget_table(rows):
    """tab_budget.tex: defended mAP under oblivious / adaptive PGD at 4/255
    and 7/255, first 300 images. The undefended row is the oblivious attack
    on the bare model (the same images and seed in every run of a model, so
    the runs must agree; the check prints any disagreement). An ensemble's
    adaptive cell is the worst case over the objectives that were run."""
    if not any(r[1] == 4 for r in rows):
        print("  no 4/255 runs yet: tab_budget.tex not written")
        return
    cell = {}
    undef = {}
    for model, e, target, obj, h in rows:
        undef.setdefault((model, e), []).append(
            h["oblivious_attacked_undefended_mAP"])
        c = cell.setdefault((model, target, e), {"obl": [], "adp": []})
        c["obl"].append(h["oblivious_attacked_defended_mAP"])
        c["adp"].append(h["adaptive_attacked_defended_mAP"])
    for k, v in sorted(undef.items()):
        print(f"  undefended oblivious {k}: {['%.4f' % x for x in v]}"
              + ("" if max(v) - min(v) < 1e-9 else "  (runs DISAGREE)"))
    lines = []
    for model, name, targets in (
            ("yolo", "\\yw{}", ("median", "jpeg", "gaussian",
                                "ens_jpeg_median_gaussian")),
            ("florence", "\\fl{}", ("median", "ens_jpeg_median_gaussian"))):
        u = [f3(statistics.fmean(undef[(model, e)])) if (model, e) in undef
             else "--" for e in (4, 7)]
        lines.append(f"{name} & no defense & {u[0]} & {u[0]} & {u[1]} & "
                     f"{u[1]} \\\\")
        for t in targets:
            cells = []
            for e in (4, 7):
                c = cell.get((model, t, e))
                cells += (["--", "--"] if c is None else
                          [f3(statistics.fmean(c["obl"])), f3(min(c["adp"]))])
            lines.append(f" & {LABEL[t]} & " + " & ".join(cells) + " \\\\")
        if model == "yolo":
            lines.append("\\midrule")
    write("tab_budget.tex", lines)


if __name__ == "__main__":
    main()
