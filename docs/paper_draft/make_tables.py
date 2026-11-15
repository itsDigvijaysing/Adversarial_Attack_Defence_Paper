#!/usr/bin/env python3
"""Regenerate the N=1000 confidence-interval tables of the draft.

    python docs/paper_draft/make_tables.py

Reads results/phase4/rescore_phase3_v2/summary_rescore_phase3_v2_latest.json
(committed; produced on CPU by phase4/p4_rescore.py) and writes
tab_ci_yolo.tex and tab_ci_florence.tex next to this file. No number in those
tables is typed by hand. Stdlib only.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "results", "phase4", "rescore_phase3_v2",
                   "summary_rescore_phase3_v2_latest.json")


def cell(d):
    """'+0.044 [+0.037, +0.052]', bold when the 95% CI excludes zero."""
    s = f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    s = s.replace("-", "$-$").replace("+", "$+$")
    return f"\\textbf{{{s}}}" if d["excludes_0"] else s


def main():
    data = json.load(open(SRC))
    for model in ("yolo", "florence"):
        cost = {r["defense"]: r for r in data["clean_cost"][model]}
        rec = {a: {r["defense"]: r for r in data["recovery"][model][a]["defenses"]}
               for a in ("fgsm", "pgd", "patch")}
        lines = []
        clean = data["clean_point"][model]["mAP"]
        atk = [data["recovery"][model][a]["attacked_mAP"]
               for a in ("fgsm", "pgd", "patch")]
        lines.append("Undefended mAP & %.3f & %.3f & %.3f & %.3f \\\\"
                     % (clean, *atk))
        lines.append("\\midrule")
        seen_ens = False
        for name, row in cost.items():
            if row["kind"] == "ensemble" and not seen_ens:
                lines.append("\\midrule")
                seen_ens = True
            cells = [cell(row["delta"]["mAP"])]
            cells += [cell(rec[a][name]["delta"]["mAP"])
                      for a in ("fgsm", "pgd", "patch")]
            lines.append(row["label"] + " & " + " & ".join(cells) + " \\\\")
        out = os.path.join(HERE, f"tab_ci_{model}.tex")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
