#!/usr/bin/env python3
"""Regenerate the data figure of the draft.

    python docs/paper_draft/make_figures.py

Reads results/phase4/rescore_phase3_v2/summary_rescore_phase3_v2_latest.json
(the same committed file as make_tables.py) and writes fig_cost_vs_patch.pdf
next to this file: for each defense, its clean cost against its mAP change
under the patch, both with 95% paired bootstrap intervals. No number is typed
by hand. Needs matplotlib; no torch, no images, no GPU.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "results", "phase4", "rescore_phase3_v2",
                   "summary_rescore_phase3_v2_latest.json")
OUT = os.path.join(HERE, "fig_cost_vs_patch.pdf")

# Two categorical slots, validated for colour-vision deficiency; the marker
# shape repeats the distinction so it never rests on colour alone.
KINDS = {"solo": ("#2a78d6", "o", "single transform"),
         "ensemble": ("#eb6834", "s", "ensemble")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d2"
TITLES = {"yolo": "YOLO-World", "florence": "Florence-2 detection"}
# Label offsets in points (dx, dy, horizontal alignment), set by eye so that
# no label sits on a marker or an error bar. Unlisted points get DEFAULT.
DEFAULT = (6, 5, "left")
OFFSETS = {
    ("yolo", "JPEG"): (6, -10, "left"),
    ("yolo", "TVM"): (-6, -10, "right"),
    ("yolo", "Median"): (-9, -11, "right"),
    ("yolo", "Gaussian"): (6, -10, "left"),
    ("yolo", "EnsBTC"): (-8, 9, "right"),
    ("yolo", "EnsJMG"): (9, -4, "left"),
    ("yolo", "EnsJMT"): (7, 7, "left"),
    ("florence", "TVM"): (6, -10, "left"),
    ("florence", "Blur+TVM"): (6, 5, "left"),
    ("florence", "Gaussian"): (6, 5, "left"),
}
# The low-cost defenses of Florence-2 sit within 0.005 mAP of each other, so
# they are named once as a group instead of five overlapping labels.
GROUPED = {"florence": {"JPEG", "Median", "EnsBTC", "EnsJMG", "EnsJMT"}}
GROUP_NOTE = {"florence": "JPEG, Median and\nthe three ensembles"}


def main():
    data = json.load(open(SRC))
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for ax, model in zip(axes, ("yolo", "florence")):
        cost = {r["defense"]: r for r in data["clean_cost"][model]}
        patch = {r["defense"]: r
                 for r in data["recovery"][model]["patch"]["defenses"]}
        lo_lim = min(min(c["delta"]["mAP"]["lo"] for c in cost.values()),
                     min(p["delta"]["mAP"]["lo"] for p in patch.values()))
        lo_lim = 0.01 * (int(lo_lim * 100) - 1)
        hi_lim = 0.03
        step = 0.02 if lo_lim > -0.09 else 0.04
        ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color=MUTED, lw=0.8,
                ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        for name, c in cost.items():
            x, y = c["delta"]["mAP"], patch[name]["delta"]["mAP"]
            color, marker, _ = KINDS[c["kind"]]
            ax.errorbar(
                x["point"], y["point"],
                xerr=[[x["point"] - x["lo"]], [x["hi"] - x["point"]]],
                yerr=[[y["point"] - y["lo"]], [y["hi"] - y["point"]]],
                fmt=marker, ms=4.5, color=color, mec="white", mew=0.6,
                ecolor=color, elinewidth=0.7, alpha=0.95, zorder=3)
            if c["label"] in GROUPED.get(model, ()):
                continue
            dx, dy, ha = OFFSETS.get((model, c["label"]), DEFAULT)
            ax.annotate(c["label"], (x["point"], y["point"]),
                        xytext=(dx, dy), textcoords="offset points",
                        ha=ha, fontsize=7, color=INK, zorder=4)
        if model in GROUP_NOTE:
            ax.annotate(GROUP_NOTE[model], (-0.008, 0.003),
                        xytext=(-0.070, 0.012), textcoords="data",
                        ha="center", va="center", fontsize=7, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                        shrinkA=2, shrinkB=12))
        mid = lo_lim + 0.30 * (hi_lim - lo_lim)
        ax.annotate("equal loss", (mid, mid), xytext=(4, -6),
                    textcoords="offset points", fontsize=7, color=MUTED,
                    rotation=45, rotation_mode="anchor", ha="center",
                    va="top")
        ax.set_xlim(lo_lim, hi_lim)
        ax.set_ylim(lo_lim, hi_lim)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.set_title(TITLES[model], fontsize=9, loc="left")
        ax.set_xlabel("clean cost (mAP change on clean images)")
        ax.tick_params(length=2.5, width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("mAP change under the patch\n(defended minus attacked)")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", ms=4.5,
                          mec="white", mew=0.6, label=lab)
               for c, m, lab in KINDS.values()]
    axes[0].legend(handles=handles, loc="lower right", frameon=False,
                   fontsize=7, handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
