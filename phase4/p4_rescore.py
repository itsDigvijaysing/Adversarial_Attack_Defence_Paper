#!/usr/bin/env python3
"""
p4_rescore.py: re-score the committed Phase-3 v2 detection dumps on CPU.

WHY
---
Reviewer 8jGZ asked for (1) the clean-image cost of every defense and (2) some
measure of uncertainty ("only one run per condition"). Both can be answered
without a GPU: the Phase-3 v2 runs left COCO-format detection dumps for every
condition in results/results_phase3_{yolo,florence}_{fgsm,pgd,patch}_v2/
(clean.json, clean_<def>.json, <attack tag>.json, <attack tag>_<def>.json for
5 solos + 3 ensembles at N=1000). This script loads NO model and NO image; it
reads those dumps plus annotations/instances_val2017.json and nothing else.

WHAT IT COMPUTES
----------------
(a) mAP@[.5:.95] and AP50 per condition with pycocotools (the same COCOeval
    call Phase 3 made), checked against the committed summary.json of each run
    directory (tolerance --tolerance, default 1e-4). Mismatches are reported in
    the log, the JSON and the markdown, and make the exit code 2.
(b) Clean cost per model: mAP(clean_<def>) - mAP(clean) for every defense.
(c) Paired image-level bootstrap, 95% percentile CIs (--B resamples, seeded):
      defended - attacked   per model x attack x defense
      clean cost            per model x defense
      attacked - clean      (attack damage)
      best ensemble - Median, under each attack (the paper's headline claim
                            that only ensembles are positive everywhere)
(d) Score-free detection quality at IoU >= 0.5: micro precision / recall / F1
    from a one-to-one greedy matching per image and class. Florence-2's scores
    are geometric pseudo-scores (box size and centring, phase-3
    `_compute_score`), so its mAP ranks boxes by geometry; these metrics do not
    use the score except to break exact IoU ties.

IMAGE SET
---------
Phase 3 scored `sorted(os.listdir(val2017))[:1000]`. COCO val2017 file names are
zero-padded image ids and every file is in instances_val2017.json, so that is
`sorted(GT image ids)[:1000]`, which is what this script uses (--n). The run
checks that rule against the dumps themselves: every detection's image id must
fall inside the scored set (reported as image_id_check; the YOLO dumps, which
keep boxes down to conf 0.001, cover exactly those 1000 ids). Images with no
detection are still evaluated (as in Phase 3: COCOeval.params.imgIds is the
full list), so a Florence dump that covers only ~980 ids is scored on 1000.

BOOTSTRAP METHOD (documented because it is the part a reviewer will check)
---------------------------------------------------------------------------
Resampling unit = image. One (B x N) matrix of image positions is drawn with
replacement from numpy's default_rng(--seed) and shared by EVERY condition, so
a difference of two conditions uses the same resampled images in both terms
(paired bootstrap). For each condition:

  1. COCOeval.evaluate() runs ONCE on the full id list. Its `evalImgs` is a
     flat list ordered [category][area range][image]; each entry holds that
     (image, category)'s score-sorted detections, their per-IoU-threshold
     matches and ignore flags, and the GT ignore flags. Matching is per image,
     so a resample never needs a re-match.
  2. For a resample (a list of N image positions WITH duplicates), evalImgs is
     rebuilt in the resampled order (a duplicated image contributes its entry
     twice) and accumulate() is called with params.imgIds = the duplicated id
     list. This is exactly what COCOeval would produce on a dataset in which
     the duplicated images were real copies (tested in tests/test_rescore.py).
  3. mAP / AP50 are read from the precision array the way summarize() does
     (area 'all', maxDets 100, mean over entries > -1).

pycocotools' accumulate() walks every detection in a Python loop (the
precision envelope) for 4 area ranges x 3 maxDets; B=1000 x 72 conditions of
that is hours. `--engine fast` (default) therefore runs a vectorised replica of
accumulate() for area 'all' / maxDets 100 only, on arrays extracted from
evalImgs once per condition. It reproduces pycocotools' operations step for
step (same concatenation order, same stable mergesort on -score, same cumsum,
same envelope, same searchsorted on the recall thresholds), so it is equal to
accumulate() up to float rounding. That is ASSERTED at run time: per condition
the replica must match COCOeval.summarize()'s stats[0] / stats[1] on the full
set, and pycocotools' own accumulate() on the first --verify-resamples
duplicated-id resamples (max |difference| is reported). `--engine
pycocotools` calls accumulate() for every resample instead (slow, same
numbers).

The bootstrap mean is compared with the point estimate for every condition
(bias, and bias in SE units, flagged above --bias-flag SE) AND for every
paired difference the tables report. mAP is not a mean over images, so a
single condition's bootstrap mAP need not be centred on the point estimate;
on these dumps it sits about one SE above it. The run measures where that
comes from, per condition (`bias_decomposition` in the JSON):
  composition   bootstrap mAP minus the same mAP with every class that is
                absent from a resample put back at its full-sample AP (the
                effect of classes dropping out of the class mean);
  within-class  the rest: each class's own AP is biased upward under
                resampling. Its envelope share is measured on the first
                --diag-resamples resamples by recomputing AP without
                accumulate()'s running-maximum precision envelope.
The bias of a class mean is the mean of the per-class biases, while its SE
shrinks with averaging over ~80 classes, which is how a small per-class drift
reaches about one SE of mAP. The bias largely cancels in a paired difference, which is what
every table reports. As a robustness check every difference also carries a
bias-corrected (BC, Efron 1981) percentile CI, and the report counts the
"excludes 0" verdicts that change under it.

What the CIs are NOT: they capture which 1000 images were scored, not attack
randomness (PGD random start) or model nondeterminism; each condition is still
one run. "Best ensemble" is chosen on the full-sample point estimate and kept
fixed inside the bootstrap, which favours it slightly. These are the N=1000
Phase-3 v2 dumps, not the N=5000 runs behind the paper's Tables 3-5 (no
N=5000 detection dump is in the repo), so the CIs describe the 1000-image
subset; its point estimates differ from the paper's.

SCORE-FREE MATCHING (d)
-----------------------
Per image and class: all (detection, non-crowd GT) pairs with IoU >= --sf-iou
are sorted by IoU (descending), then score (descending, tie-break only), then
detection and GT index, and taken greedily one-to-one. Unmatched detections
whose intersection-over-detection-area with a crowd GT of the same class is
>= --sf-iou are ignored (COCO's crowd rule); the rest are false positives.
Unmatched non-crowd GT are false negatives. P / R / F1 are pooled over all
images and classes (micro). Florence: every predicted box (its pipeline has no
score threshold). YOLO: twice, every dumped box (conf >= 0.001, the Phase-3
predict threshold) and conf >= --yolo-sf-conf (default 0.25, ultralytics'
default predict threshold), because at 0.001 precision is dominated by
low-confidence boxes that mAP discounts but a count metric cannot. Count
metrics bootstrap by weighting each image by its multiplicity in the resample.

OUTPUT (repo artifact policy, see CLAUDE.md "Phase-4 conventions")
--------------------------------------------------------------------
results/phase4/rescore_phase3_v2/
    summary_rescore_phase3_v2_<timestamp>.json / .md   immutable
    summary_rescore_phase3_v2_latest.json / .md        mirror of the newest run
    run.log                                            appended
No dump is written anywhere: pycocotools' loadRes() takes the list in memory.
A subset run (--only / --models / --attacks) is marked PARTIAL and goes to
<output>/smoke_test/ (gitignored wholesale) unless --output-dir is given.

USAGE
-----
    /home/king/miniconda3/envs/vlm_ftune/bin/python phase4/p4_rescore.py
    ... --B 200 --workers 4                       # fewer resamples / processes
    ... --models yolo --attacks pgd --only clean --B 50 --output-dir /tmp/x
    ... --list                                    # show discovered conditions
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

PHASE4_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PHASE4_DIR)
if PHASE4_DIR not in sys.path:
    sys.path.insert(0, PHASE4_DIR)

from p4_logging import artifact_report, is_committed, run_log  # noqa: E402

EXPERIMENT = "rescore_phase3_v2"
STEM = f"summary_{EXPERIMENT}"
DEFAULT_INPUT_ROOT = os.path.join(REPO_ROOT, "results")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "phase4", EXPERIMENT)
INPUT_DIR_PATTERN = "results_phase3_{model}_{attack}_v2"

MODELS = ("yolo", "florence")
ATTACKS = ("fgsm", "pgd", "patch")
MODEL_LABEL = {"yolo": "YOLOv8x-worldv2", "florence": "Florence-2-base"}
ATTACK_LABEL = {"fgsm": "FGSM", "pgd": "PGD", "patch": "Patch"}
# Paper labels; mirrors p4_core.SOLO_PAPER_NAMES / ENSEMBLE_PAPER_NAMES (not
# imported: p4_core pulls in more than this CPU-only script needs).
DEFENSE_LABEL = {
    "jpeg": "JPEG", "median": "Median", "gaussian": "Gaussian", "tvm": "TVM",
    "blur_tvm": "Blur+TVM",
    "ens_jpeg_median_gaussian": "EnsJMG", "ens_jpeg_median_tvm": "EnsJMT",
    "ens_blur_tvm_combo": "EnsBTC",
}
MEDIAN_KEY = "median"
METRICS = ("mAP", "AP50")


def dlabel(name):
    return DEFENSE_LABEL.get(name, name)


@contextlib.contextmanager
def quiet():
    """Swallow pycocotools' prints (log hygiene: one [eval] line per condition)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def ids_digest(ids) -> str:
    """Same digest as p4_core.ids_digest, so image sets can be compared."""
    return hashlib.sha1(json.dumps(sorted(ids)).encode()).hexdigest()[:16]


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ======================================================================
# 1. CLI and output paths (stdlib + numpy only: importable without pycocotools)
# ======================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="Re-score the Phase-3 v2 detection dumps on CPU: mAP "
                    "check vs summary.json, clean cost, paired image-level "
                    "bootstrap CIs, score-free P/R/F1. Loads no model/image.")
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--attacks", nargs="+", choices=ATTACKS,
                   default=list(ATTACKS))
    p.add_argument("--only", nargs="+", default=None, metavar="TAG",
                   help="Score only these condition tags (e.g. clean "
                        "pgd_eps0.03 clean+median). Marks the run PARTIAL.")
    p.add_argument("--B", type=int, default=1000,
                   help="Bootstrap resamples (default 1000).")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed of the shared resample matrix.")
    p.add_argument("--ci", type=float, default=95.0,
                   help="Percentile CI level in %% (default 95).")
    p.add_argument("--n", type=int, default=1000,
                   help="Images: the first n sorted val2017 ids (Phase 3: "
                        "1000).")
    p.add_argument("--input-root", default=DEFAULT_INPUT_ROOT,
                   help="Directory holding results_phase3_<model>_<attack>_v2/.")
    p.add_argument("--ann-file", default=None,
                   help="instances_val2017.json (default: auto-resolve).")
    p.add_argument("--output-dir", default=None,
                   help=f"Default {os.path.relpath(DEFAULT_OUTPUT_DIR, REPO_ROOT)}"
                        "/ (a PARTIAL run: its smoke_test/ subdir).")
    p.add_argument("--workers", type=int, default=None,
                   help="Processes, one condition each (default "
                        "min(6, cpus-2)). 1 = run inline.")
    p.add_argument("--task-timeout", type=float, default=1800.0,
                   help="Seconds to wait for the next finished condition "
                        "before aborting (guards against a stuck worker).")
    p.add_argument("--engine", choices=["fast", "pycocotools"], default="fast",
                   help="fast = vectorised replica of accumulate(), verified "
                        "against it at run time; pycocotools = call "
                        "accumulate() for every resample (slow).")
    p.add_argument("--verify-resamples", type=int, default=3,
                   help="Resamples per condition cross-checked against "
                        "pycocotools' accumulate() (default 3).")
    p.add_argument("--tolerance", type=float, default=1e-4,
                   help="Allowed |recomputed - summary.json| (default 1e-4).")
    p.add_argument("--sf-iou", type=float, default=0.5,
                   help="IoU threshold of the score-free metrics.")
    p.add_argument("--yolo-sf-conf", type=float, default=0.25,
                   help="Extra YOLO confidence cut for the score-free metrics.")
    p.add_argument("--bias-flag", type=float, default=0.5,
                   help="Flag |bootstrap mean - point| above this many SEs.")
    p.add_argument("--diag-resamples", type=int, default=200,
                   help="Resamples per condition used to measure the "
                        "envelope share of the bootstrap bias (default 200; "
                        "0 = skip).")
    p.add_argument("--list", action="store_true",
                   help="List the discovered conditions and exit.")
    return p


def finalize_args(args):
    if args.B < 1:
        raise SystemExit("--B must be >= 1")
    if not 50.0 < args.ci < 100.0:
        raise SystemExit("--ci must be in (50, 100)")
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    if args.diag_resamples < 0:
        raise SystemExit("--diag-resamples must be >= 0")
    if args.task_timeout <= 0:
        raise SystemExit("--task-timeout must be > 0")
    args.run_id = time.strftime("%Y%m%d-%H%M%S")
    args.partial = bool(args.only) or (
        set(args.models) != set(MODELS) or set(args.attacks) != set(ATTACKS))
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR
        if args.partial:
            args.output_dir = os.path.join(args.output_dir, "smoke_test")
    if args.workers is None:
        args.workers = max(1, min(6, (os.cpu_count() or 2) - 2))
    args.workers = max(1, args.workers)
    return args


def parse_args(argv=None):
    return finalize_args(build_parser().parse_args(argv))


def output_paths(output_dir, run_id):
    """Every file this script writes. All but run.log start with 'summary'."""
    j = lambda name: os.path.join(output_dir, name)  # noqa: E731
    return {
        "json": j(f"{STEM}_{run_id}.json"),
        "json_latest": j(f"{STEM}_latest.json"),
        "md": j(f"{STEM}_{run_id}.md"),
        "md_latest": j(f"{STEM}_latest.md"),
        "log": j("run.log"),
    }


def resolve_ann_file(primary):
    candidates = [primary] if primary else []
    candidates += [
        os.path.join(REPO_ROOT, "annotations", "instances_val2017.json"),
        os.path.join(REPO_ROOT, "Dataset", "annotations",
                     "instances_val2017.json")]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise SystemExit("instances_val2017.json not found. Tried: "
                     + ", ".join(candidates))


# ======================================================================
# 2. Input discovery
# ======================================================================
def tag_to_file(tag):
    """phase3_common.evaluate_all_conditions' dump name for a condition tag."""
    return tag.replace("+", "_").replace("/", "_") + ".json"


def discover(input_root, model, attack):
    """Read one run directory: attack tag, defenses (summary order), files."""
    d = os.path.join(input_root, INPUT_DIR_PATTERN.format(model=model,
                                                          attack=attack))
    spath = os.path.join(d, "summary.json")
    if not os.path.isfile(spath):
        raise SystemExit(f"missing {spath}")
    with open(spath) as f:
        summary = json.load(f)
    atag = summary["attack"]
    defenses = [(name, str(v.get("kind", "solo")).lower())
                for name, v in summary["defenses"].items()]
    tags = (["clean"] + [f"clean+{n}" for n, _ in defenses]
            + [atag] + [f"{atag}+{n}" for n, _ in defenses])
    files, missing = {}, []
    for t in tags:
        path = os.path.join(d, tag_to_file(t))
        (files.__setitem__(t, path) if os.path.isfile(path)
         else missing.append(path))
    if missing:
        raise SystemExit("missing dumps:\n  " + "\n  ".join(missing))
    return {"model": model, "attack": attack, "dir": d, "summary": summary,
            "attack_tag": atag, "defenses": defenses, "files": files}


# ======================================================================
# 3. Ground truth (reduced to the scored ids: identical COCOeval result,
#    because _prepare() only loads GT of params.imgIds anyway)
# ======================================================================
def load_gt(ann_file, n):
    from pycocotools.coco import COCO
    with open(ann_file) as f:
        full = json.load(f)
    all_ids = sorted(im["id"] for im in full["images"])
    ids = all_ids[:n]
    keep = set(ids)
    ds = {k: v for k, v in full.items() if k not in ("images", "annotations")}
    ds["images"] = [im for im in full["images"] if im["id"] in keep]
    ds["annotations"] = [a for a in full["annotations"]
                         if a["image_id"] in keep]
    del full
    gt = COCO()
    gt.dataset = ds
    with quiet():
        gt.createIndex()
    return gt, ids, len(all_ids)


def gt_by_img_cat(anns):
    """{(image_id, category_id): (non-crowd xywh (G,4), crowd xywh (C,4))}."""
    tmp = {}
    for a in anns:
        key = (a["image_id"], a["category_id"])
        slot = tmp.setdefault(key, ([], []))
        slot[1 if a.get("iscrowd", 0) else 0].append(a["bbox"])
    return {k: (np.asarray(g, float).reshape(-1, 4),
                np.asarray(c, float).reshape(-1, 4))
            for k, (g, c) in tmp.items()}


# ======================================================================
# 4. Score-free matching (d)
# ======================================================================
def _xyxy(b):
    return b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]


def box_iou_xywh(a, b):
    """IoU matrix (len(a), len(b)) for COCO xywh boxes (no +1, as pycocotools)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) -
                 np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) -
                 np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def box_iof_xywh(dets, crowd):
    """Intersection over DETECTION area (pycocotools' IoU for a crowd GT)."""
    d = np.asarray(dets, float).reshape(-1, 4)
    c = np.asarray(crowd, float).reshape(-1, 4)
    dx1, dy1, dx2, dy2 = _xyxy(d)
    cx1, cy1, cx2, cy2 = _xyxy(c)
    iw = np.clip(np.minimum(dx2[:, None], cx2[None]) -
                 np.maximum(dx1[:, None], cx1[None]), 0, None)
    ih = np.clip(np.minimum(dy2[:, None], cy2[None]) -
                 np.maximum(dy1[:, None], cy1[None]), 0, None)
    inter = iw * ih
    area = (d[:, 2] * d[:, 3])[:, None]
    return np.divide(inter, np.broadcast_to(area, inter.shape),
                     out=np.zeros_like(inter), where=area > 0)


def match_greedy(det_boxes, det_scores, gt_boxes, crowd_boxes=None,
                 iou_thr=0.5):
    """One-to-one greedy matching of one image x class, ordered by IoU.

    Returns (pairs, ignored): pairs = [(det_idx, gt_idx)] in match order;
    ignored = bool mask of unmatched detections covered by a crowd GT.
    Score only breaks exact IoU ties (then det index, then GT index).
    """
    det_boxes = np.asarray(det_boxes, float).reshape(-1, 4)
    det_scores = np.asarray(det_scores, float).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    D, G = len(det_boxes), len(gt_boxes)
    pairs = []
    det_free = np.ones(D, bool)
    if D and G:
        iou = box_iou_xywh(det_boxes, gt_boxes)
        di, gi = np.nonzero(iou >= iou_thr)
        if len(di):
            order = np.lexsort((gi, di, -det_scores[di], -iou[di, gi]))
            gt_free = np.ones(G, bool)
            for o in order:
                d, g = int(di[o]), int(gi[o])
                if det_free[d] and gt_free[g]:
                    det_free[d] = gt_free[g] = False
                    pairs.append((d, g))
    ignored = np.zeros(D, bool)
    if crowd_boxes is not None and len(crowd_boxes) and det_free.any():
        iof = box_iof_xywh(det_boxes, crowd_boxes)
        ignored = det_free & (iof.max(axis=1) >= iou_thr)
    return pairs, ignored


def match_counts(det_boxes, det_scores, gt_boxes, crowd_boxes=None,
                 iou_thr=0.5):
    """(tp, fp, fn, ignored) for one image x class."""
    pairs, ignored = match_greedy(det_boxes, det_scores, gt_boxes,
                                  crowd_boxes, iou_thr)
    D = len(np.asarray(det_scores).reshape(-1))
    G = len(np.asarray(gt_boxes).reshape(-1, 4))
    tp = len(pairs)
    n_ign = int(ignored.sum())
    return tp, D - tp - n_ign, G - tp, n_ign


def score_free_counts(dets, gt_sf, img_ids, iou_thr=0.5, score_thr=None):
    """Per-image TP / FP / FN / ignored counts (arrays aligned with img_ids)."""
    pos = {iid: i for i, iid in enumerate(img_ids)}
    N = len(img_ids)
    out = {k: np.zeros(N, np.int64) for k in ("tp", "fp", "fn", "ignored")}
    grouped = {}
    for d in dets:
        if d["image_id"] not in pos:
            continue
        if score_thr is not None and d["score"] < score_thr:
            continue
        slot = grouped.setdefault((d["image_id"], d["category_id"]), ([], []))
        slot[0].append(d["bbox"])
        slot[1].append(d["score"])
    empty4 = np.zeros((0, 4))
    keys = set(grouped) | {k for k in gt_sf if k[0] in pos}
    for key in keys:
        g, c = gt_sf.get(key, (empty4, empty4))
        boxes, scores = grouped.get(key, (empty4, []))
        tp, fp, fn, ign = match_counts(boxes, scores, g, c, iou_thr)
        i = pos[key[0]]
        out["tp"][i] += tp
        out["fp"][i] += fp
        out["fn"][i] += fn
        out["ignored"][i] += ign
    return out


def prf(tp, fp, fn):
    """Micro precision / recall / F1 from (possibly array-valued) totals."""
    tp, fp, fn = (np.asarray(x, float) for x in (tp, fp, fn))
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp),
                  where=(2 * tp + fp + fn) > 0)
    return p, r, f


def sf_thresholds(model, yolo_conf):
    if model == "yolo":
        return [("all", None), (f"conf{yolo_conf:g}", float(yolo_conf))]
    return [("all", None)]


def sf_label(key, model):
    if key == "all":
        return ("all dumped boxes (conf >= 0.001)" if model == "yolo"
                else "all predicted boxes")
    return f"conf >= {key[4:]}"


# ======================================================================
# 5. mAP: pycocotools point estimate, literal resample, fast replica
# ======================================================================
def summarize_precision(prec):
    """(mAP@[.5:.95], AP50) from a (T, R, K) precision array, as summarize()."""
    s = prec[prec > -1]
    m = float(np.mean(s)) if s.size else -1.0
    s50 = prec[0][prec[0] > -1]
    a50 = float(np.mean(s50)) if s50.size else -1.0
    return m, a50


def _area_all_index(params):
    return list(params.areaRngLbl).index("all")


def literal_resample(ev, pos):
    """pycocotools' own accumulate() on evalImgs rebuilt for a duplicated id list.

    `pos` = image positions (into ev._paramsEval.imgIds), duplicates allowed.
    Only area 'all' and maxDets[-1] are accumulated (what stats[0:2] read).
    """
    pe = ev._paramsEval
    I0, A0, K0 = len(pe.imgIds), len(pe.areaRng), len(pe.catIds)
    a = _area_all_index(pe)
    new_pe = copy.deepcopy(pe)
    new_pe.imgIds = [pe.imgIds[int(i)] for i in pos]
    new_pe.areaRng = [pe.areaRng[a]]
    new_pe.areaRngLbl = ["all"]
    new_pe.maxDets = [pe.maxDets[-1]]
    shadow = copy.copy(ev)
    shadow._paramsEval = new_pe
    shadow.evalImgs = [ev.evalImgs[k * A0 * I0 + a * I0 + int(i)]
                       for k in range(K0) for i in pos]
    with quiet():
        shadow.accumulate(copy.deepcopy(new_pe))
    return summarize_precision(shadow.eval["precision"][:, :, :, 0, 0])


def compact_from_evalimgs(eval_imgs, n_cats, n_areas, n_imgs, area_index,
                          n_iou, max_det, rec_thrs):
    """Per-category flat arrays of what accumulate() reads from evalImgs.

    For category k: S (scores, image-major, each image's dets in evalImgs
    order), TP / FP (n_iou x D bool, exactly accumulate's tps / fps), and per
    image start / length / count of non-ignored GT. None for a category with
    no non-ignored GT anywhere (accumulate skips it in every resample).
    """
    cats = []
    for k in range(n_cats):
        base = k * n_areas * n_imgs + area_index * n_imgs
        lens = np.zeros(n_imgs, np.int64)
        npig = np.zeros(n_imgs, np.int64)
        S, TP, FP = [], [], []
        for i in range(n_imgs):
            e = eval_imgs[base + i]
            if e is None:
                continue
            s = np.asarray(e["dtScores"][0:max_det], float)
            dtm = np.asarray(e["dtMatches"])[:, 0:max_det]
            dtig = np.asarray(e["dtIgnore"])[:, 0:max_det]
            lens[i] = len(s)
            npig[i] = np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)
            S.append(s)
            TP.append(np.logical_and(dtm, np.logical_not(dtig)))
            FP.append(np.logical_and(np.logical_not(dtm),
                                     np.logical_not(dtig)))
        if npig.sum() == 0:
            cats.append(None)
            continue
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        cats.append({
            "S": np.concatenate(S) if S else np.zeros(0),
            "TP": (np.concatenate(TP, axis=1) if TP
                   else np.zeros((n_iou, 0), bool)),
            "FP": (np.concatenate(FP, axis=1) if FP
                   else np.zeros((n_iou, 0), bool)),
            "starts": starts, "lens": lens, "npig": npig})
    return {"T": n_iou, "R": len(rec_thrs), "K": n_cats,
            "rec_thrs": np.asarray(rec_thrs, float), "cats": cats}


def compact_from_eval(ev):
    pe = ev._paramsEval
    return compact_from_evalimgs(
        ev.evalImgs, len(pe.catIds), len(pe.areaRng), len(pe.imgIds),
        _area_all_index(pe), len(pe.iouThrs), pe.maxDets[-1], pe.recThrs)


def fast_precision(compact, pos, envelope=True):
    """Vectorised accumulate() for one resample: (T, R, K) precision array.

    envelope=False skips accumulate()'s running-maximum precision envelope
    (a diagnostic only: that is not COCO AP).
    """
    T, R, K = compact["T"], compact["R"], compact["K"]
    rec = compact["rec_thrs"]
    prec = -np.ones((T, R, K))
    pos = np.asarray(pos, np.int64)
    for k, c in enumerate(compact["cats"]):
        if c is None:
            continue
        npig = int(c["npig"][pos].sum())
        if npig == 0:
            continue
        lens = c["lens"][pos]
        nd = int(lens.sum())
        if nd == 0:
            prec[:, :, k] = 0.0
            continue
        offs = np.cumsum(lens) - lens
        gather = np.repeat(c["starts"][pos] - offs, lens) + np.arange(nd)
        order = np.argsort(-c["S"][gather], kind="mergesort")
        g = gather[order]
        tp = np.cumsum(c["TP"][:, g], axis=1).astype(float)
        fp = np.cumsum(c["FP"][:, g], axis=1).astype(float)
        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))
        if envelope:
            pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
        for t in range(T):
            inds = np.searchsorted(rc[t], rec, side="left")
            ok = inds < nd
            q = np.zeros(R)
            q[ok] = pr[t, inds[ok]]
            prec[t, :, k] = q
    return prec


def fast_resample(compact, pos):
    return summarize_precision(fast_precision(compact, pos))


def class_ap(prec):
    """Per-class AP over all IoU and recall thresholds; NaN = class absent.

    accumulate() writes a class's whole (T, R) slice or leaves it at -1, so
    mean over classes of this == summarize_precision(prec)[0].
    """
    present = (prec > -1).all(axis=(0, 1))
    return np.where(present, prec.mean(axis=(0, 1)), np.nan)


def filled_map(ap_b, ap_point):
    """mAP of a resample with absent classes put back at their full-sample AP.

    Classes absent from the full sample stay out (they are absent from every
    resample). mAP(resample) - filled_map isolates the class-composition part
    of the bootstrap bias.
    """
    keep = ~np.isnan(ap_point)
    if not keep.any():
        return -1.0
    a = np.where(np.isnan(ap_b), ap_point, ap_b)[keep]
    return float(np.mean(a))


# ======================================================================
# 6. One condition (runs in a worker process)
# ======================================================================
_CTX: dict = {}


def evaluate_condition(task, ctx=None):
    ctx = ctx or _CTX
    from pycocotools.cocoeval import COCOeval
    t0 = time.time()
    with open(task["path"]) as f:
        dets = json.load(f)
    ids, idset, idx = ctx["ids"], ctx["idset"], ctx["idx"]
    n_raw = len(dets)
    present = sorted({d["image_id"] for d in dets})
    dets = [d for d in dets if d["image_id"] in idset]
    N, B = len(ids), idx.shape[0]

    sf = {key: score_free_counts(dets, ctx["gt_sf"], ids, ctx["sf_iou"], thr)
          for key, thr in sf_thresholds(task["model"], ctx["yolo_sf_conf"])}

    res = {"key": task["key"], "model": task["model"], "n_dets_raw": n_raw,
           "n_dets_outside_ids": n_raw - len(dets), "n_dets": len(dets),
           "image_ids_present": present, "score_free_counts": sf}
    if not dets:  # Phase 3's coco_eval_tag returned zeros(12) here
        res.update(stats=[0.0] * 12, point={"mAP": 0.0, "AP50": 0.0},
                   boot={m: np.zeros(B) for m in METRICS},
                   bias_decomposition=None,
                   engine_point_diff=0.0, engine_resample_diffs=[],
                   timing={"evaluate_s": 0.0, "boot_s": 0.0, "diag_s": 0.0,
                           "total_s": time.time() - t0})
        return res

    gt = ctx["gt"]
    with quiet():
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    t_eval = time.time() - t0
    stats = [float(x) for x in ev.stats]
    compact = compact_from_eval(ev)

    fm, f50 = fast_resample(compact, np.arange(N))
    point_diff = max(abs(fm - stats[0]), abs(f50 - stats[1]))
    if point_diff > 1e-9:
        raise AssertionError(
            f"{task['key']}: fast accumulate disagrees with COCOeval on the "
            f"full set ({fm} vs {stats[0]}, {f50} vs {stats[1]})")
    diffs = []
    for r in range(min(ctx["verify_resamples"], B)):
        lm, l50 = literal_resample(ev, idx[r])
        vm, v50 = fast_resample(compact, idx[r])
        diffs.append(max(abs(lm - vm), abs(l50 - v50)))
    if diffs and max(diffs) > 1e-9:
        raise AssertionError(f"{task['key']}: fast accumulate disagrees with "
                             f"pycocotools on a resample (max {max(diffs)})")

    t1 = time.time()
    ap_point = class_ap(fast_precision(compact, np.arange(N)))
    boot = np.zeros((B, 2))
    filled = np.zeros(B)
    for b in range(B):
        prec = fast_precision(compact, idx[b])
        filled[b] = filled_map(class_ap(prec), ap_point)
        boot[b] = (summarize_precision(prec) if ctx["engine"] == "fast"
                   else literal_resample(ev, idx[b]))
    t_boot = time.time() - t1
    decomp = bias_decomposition(compact, idx, stats[0], boot[:, 0], filled,
                                ap_point, ctx.get("diag_resamples", 0))
    res.update(stats=stats, point={"mAP": stats[0], "AP50": stats[1]},
               boot={"mAP": boot[:, 0], "AP50": boot[:, 1]},
               bias_decomposition=decomp,
               engine_point_diff=point_diff, engine_resample_diffs=diffs,
               timing={"evaluate_s": t_eval, "boot_s": t_boot,
                       "diag_s": time.time() - t1 - t_boot,
                       "total_s": time.time() - t0})
    return res


def bias_decomposition(compact, idx, point, boot_map, filled, ap_point,
                       n_diag):
    """Split bootstrap-mean-minus-point of one condition's mAP.

    total = composition + within_class (exact, over all B resamples).
    On the first n_diag resamples, within_class is recomputed without the
    precision envelope; envelope_share_diag = the part the envelope adds.
    """
    boot_map = np.asarray(boot_map, float)
    out = {"n_resamples": int(boot_map.size),
           "total": float(boot_map.mean() - point),
           "composition": float(boot_map.mean() - filled.mean()),
           "within_class": float(filled.mean() - point),
           "n_diag": 0}
    n_diag = int(min(n_diag, boot_map.size))
    if n_diag:
        ap0 = class_ap(fast_precision(compact, np.arange(len(idx[0])),
                                      envelope=False))
        point0 = float(np.nanmean(ap0)) if (~np.isnan(ap0)).any() else -1.0
        f0 = np.array([filled_map(class_ap(fast_precision(
            compact, idx[b], envelope=False)), ap0) for b in range(n_diag)])
        w_env = float(filled[:n_diag].mean() - point)
        w_noenv = float(f0.mean() - point0)
        out.update(n_diag=n_diag, within_class_diag=w_env,
                   within_class_no_envelope_diag=w_noenv,
                   envelope_share_diag=w_env - w_noenv,
                   point_no_envelope=point0)
    return out


def _worker(task):
    return evaluate_condition(task, _CTX)


# ======================================================================
# 7. Statistics on the bootstrap arrays
# ======================================================================
def percentile_ci(x, ci):
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(np.asarray(x, float), [alpha, 100.0 - alpha])
    return float(lo), float(hi)


def bc_percentile_ci(x, point, ci):
    """Bias-corrected (BC) percentile CI (Efron 1981), a robustness check only.

    z0 = Phi^-1(share of resamples below the point estimate, ties half);
    the percentile levels move to Phi(2 z0 +- z_{(1+ci)/2}). Equals the plain
    percentile CI when the bootstrap distribution is centred on the point.
    """
    from statistics import NormalDist
    x = np.asarray(x, float)
    B = x.size
    share = (np.count_nonzero(x < point) + 0.5 * np.count_nonzero(x == point)) / B
    share = min(max(share, 0.5 / B), 1.0 - 0.5 / B)
    nd = NormalDist()
    z0 = nd.inv_cdf(share)
    za = nd.inv_cdf(0.5 + ci / 200.0)
    lo_q, hi_q = nd.cdf(2 * z0 - za), nd.cdf(2 * z0 + za)
    lo, hi = np.percentile(x, [100.0 * lo_q, 100.0 * hi_q])
    return float(lo), float(hi)


def paired_delta(a_boot, b_boot, a_point, b_point, ci):
    """a - b on the same resamples. share_le0 = fraction of resamples <= 0.

    bias = bootstrap mean - point; bc_lo / bc_hi = bias-corrected percentile
    CI, reported to show whether a verdict depends on the bootstrap's bias.
    """
    d = np.asarray(a_boot, float) - np.asarray(b_boot, float)
    point = float(a_point - b_point)
    lo, hi = percentile_ci(d, ci)
    se = float(d.std(ddof=1)) if d.size > 1 else 0.0
    bias = float(d.mean() - point)
    bc_lo, bc_hi = bc_percentile_ci(d, point, ci)
    return {"point": point, "lo": lo, "hi": hi,
            "boot_mean": float(d.mean()), "se": se, "bias": bias,
            "bias_over_se": (bias / se) if se > 0 else 0.0,
            "share_le0": float(np.mean(d <= 0)),
            "excludes_0": bool(lo > 0 or hi < 0),
            "bc_lo": bc_lo, "bc_hi": bc_hi,
            "bc_excludes_0": bool(bc_lo > 0 or bc_hi < 0)}


def boot_summary(point, boot, ci):
    boot = np.asarray(boot, float)
    se = float(boot.std(ddof=1)) if boot.size > 1 else 0.0
    lo, hi = percentile_ci(boot, ci)
    bias = float(boot.mean() - point)
    bc_lo, bc_hi = bc_percentile_ci(boot, point, ci)
    return {"point": float(point), "mean": float(boot.mean()), "se": se,
            "lo": lo, "hi": hi, "bias": bias,
            "bias_over_se": (bias / se) if se > 0 else 0.0,
            "bc_lo": bc_lo, "bc_hi": bc_hi}


def multiplicity_matrix(idx, n):
    """(B, n) counts: how often each image appears in each resample."""
    W = np.zeros((idx.shape[0], n), np.int64)
    for b in range(idx.shape[0]):
        W[b] = np.bincount(idx[b], minlength=n)
    return W


def sf_boot(counts, W):
    """Bootstrap P / R / F1 arrays from per-image counts and multiplicities."""
    tp, fp, fn = (W @ counts[k] for k in ("tp", "fp", "fn"))
    p, r, f = prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f}


# ======================================================================
# 8. Report
# ======================================================================
def f4(v):
    return "n/a" if v is None else f"{v:.4f}"


def fd(d, nd=4):
    """'+0.0123 [+0.0050, +0.0190]' (ASCII signs) with a marker if CI spans 0.

    When the bias-corrected CI gives the other verdict, it is shown as well,
    so a reader sees which conclusions depend on the bootstrap's bias.
    """
    if d is None:
        return "n/a"
    mark = "" if d["excludes_0"] else " (CI spans 0)"
    if "bc_excludes_0" in d and d["bc_excludes_0"] != d["excludes_0"]:
        mark += (f" (BC CI [{d['bc_lo']:+.{nd}f}, {d['bc_hi']:+.{nd}f}] "
                 + ("excludes 0)" if d["bc_excludes_0"] else "spans 0)"))
    return (f"{d['point']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"
            f"{mark}")


def build_markdown(P):
    cfg = P["config"]
    L = []
    L.append("# Phase-3 v2 detection dumps, re-scored on CPU "
             f"(`{EXPERIMENT}`)\n")
    if P.get("partial"):
        L.append("> **PARTIAL RUN** (subset of models / attacks / "
                 "conditions). Do not cite.\n")
    ag = P["agreement"]
    L.append(f"- Run `{P['run_id']}`, git `{P.get('git', 'unknown')}`. No "
             "model and no image was loaded: inputs are the COCO-format "
             "dumps the Phase-3 v2 runs wrote, plus the COCO GT.")
    L.append(f"- Images: **n = {P['image_ids']['n']}**, the first "
             f"{P['image_ids']['n']} sorted val2017 ids (the Phase-3 subset), "
             f"id digest `{P['image_ids']['digest']}`.")
    L.append(f"- Bootstrap: **B = {cfg['B']}** paired image-level resamples "
             f"(seed {cfg['seed']}, one resample matrix shared by every "
             f"condition), {cfg['ci']:g}% percentile CIs, engine "
             f"`{cfg['engine']}`.")
    L.append(f"- summary.json agreement (|diff| <= {ag['tolerance']:g}): "
             f"**{ag['n_checked'] - ag['n_mismatch']}/{ag['n_checked']}** "
             f"values match, max |diff| = {ag['max_abs_diff']:.2e}.")
    for m in ag["mismatches"]:
        L.append(f"  - MISMATCH {m['model']}/{m['attack']} `{m['field']}`: "
                 f"committed {m['committed']:.6f}, recomputed "
                 f"{m['recomputed']:.6f}")
    ec = P["engine_check"]
    L.append(f"- Fast accumulate vs pycocotools: max |diff| "
             f"{ec['max_abs_diff_point']:.1e} on the full set, "
             f"{ec['max_abs_diff_resamples']:.1e} over "
             f"{ec['n_resamples_checked']} duplicated-id resamples.")
    ident = ", ".join(f"{MODEL_LABEL[m]} {'yes' if v else 'NO'}"
                      for m, v in P["clean_dumps_identical"].items())
    L.append(f"- Clean dumps byte-identical across the attack dirs: {ident}.")
    for m, c in P.get("image_id_check", {}).items():
        L.append(f"- Image ids in the {MODEL_LABEL[m]} dumps: "
                 f"{c['n_ids_in_dumps']} distinct over {c['n_dumps']} dumps, "
                 f"{'all' if c['subset_of_scored_ids'] else 'NOT all'} inside "
                 f"the scored set ({c['n_dets_outside_scored_ids']} detections "
                 "outside); images without a detection still count (their GT "
                 "are misses).")
    L.append("")

    L.append("## How to read\n")
    L.append("- Every number is mAP@[.5:.95] unless the column says AP50 or "
             "F1. Brackets are the bootstrap CI; \"(CI spans 0)\" marks a "
             "difference the 1000-image sample cannot separate from zero.")
    L.append("- **Clean cost** = mAP(defense on clean images) - mAP(clean). "
             "Negative = the defense costs accuracy when there is no attack.")
    L.append("- **Recovery** = mAP(defended) - mAP(attacked), same images in "
             "each resample. \"share <= 0\" is the fraction of resamples in "
             "which the defense did not help; it is not a p-value.")
    L.append("- The CIs cover image sampling only. Each condition is still "
             "one attack run (no seeds), so they do not capture attack or "
             "model randomness.")
    L.append("- Florence-2 boxes carry geometric pseudo-scores (box size and "
             "centring, not confidence), so its mAP ranks boxes by geometry. "
             "The score-free table uses no score beyond tie-breaking.")
    L.append("- Best ensemble is chosen on the point estimate and held fixed "
             "in the bootstrap, which favours it slightly.")
    L.append("- These are the N=1000 Phase-3 v2 dumps (5 solos, 3 ensembles), "
             "not the N=5000 runs behind the paper's Tables 3-5, whose dumps "
             "are not in the repo. Point values therefore differ from the "
             "paper's; the CIs say how much a 1000-image subset moves them.")
    L.append("")

    L.append("## Clean cost per defense\n")
    for model, rows in P["clean_cost"].items():
        L.append(f"### {MODEL_LABEL[model]} (clean mAP "
                 f"{f4(P['clean_point'][model]['mAP'])}, AP50 "
                 f"{f4(P['clean_point'][model]['AP50'])})\n")
        L.append("| Defense | Kind | mAP on clean | Clean cost, mAP "
                 "[CI] | Clean cost, AP50 [CI] |")
        L.append("|---|---|---|---|---|")
        for r in rows:
            L.append(f"| {r['label']} | {r['kind']} | {f4(r['mAP'])} | "
                     f"{fd(r['delta']['mAP'])} | {fd(r['delta']['AP50'])} |")
        L.append("")

    L.append("## Recovery under attack (defended - attacked)\n")
    for model, by_attack in P["recovery"].items():
        for attack, blk in by_attack.items():
            L.append(f"### {MODEL_LABEL[model]} x {blk['attack_label']}\n")
            L.append("| Defense | Kind | mAP | Recovery, mAP [CI] | "
                     "Recovery, AP50 [CI] | share <= 0 |")
            L.append("|---|---|---|---|---|---|")
            L.append(f"| no defense (attacked) | | {f4(blk['attacked_mAP'])} "
                     f"| damage vs clean: {fd(blk['attack_damage']['mAP'])} "
                     f"| {fd(blk['attack_damage']['AP50'])} | |")
            for r in blk["defenses"]:
                L.append(f"| {r['label']} | {r['kind']} | {f4(r['mAP'])} | "
                         f"{fd(r['delta']['mAP'])} | "
                         f"{fd(r['delta']['AP50'])} | "
                         f"{r['delta']['mAP']['share_le0']:.3f} |")
            L.append("")

    L.append("## Headline check: where is recovery positive?\n")
    L.append("The paper's headline said only ensembles have positive recovery "
             "in all nine settings (3 attacks x YOLO OD, Florence OD, "
             "Florence OCR), yet in its own N=5000 tables Median is positive "
             "in all nine too. Here: the six detection settings at N=1000 "
             "(OCR is not in these dumps), mAP recovery = defended - "
             "attacked.\n")
    L.append("| Defense | Kind | Settings | Point > 0 | CI above 0 | "
             "CI below 0 | Settings with point <= 0 |")
    L.append("|---|---|---|---|---|---|---|")
    for h in P.get("headline_check", []):
        L.append(f"| {h['label']} | {h['kind']} | {h['settings']} | "
                 f"{h['point_pos']} | {h['ci_above_0']} | {h['ci_below_0']} "
                 f"| {', '.join(h['point_le0_settings']) or '-'} |")
    L.append("")

    L.append("## Best ensemble vs Median, under attack\n")
    L.append("The paired difference behind that claim: does the best "
             "ensemble beat Median on the same resampled images?\n")
    L.append("| Model | Attack | Best ensemble | Ensemble mAP | Median mAP | "
             "Ensemble - Median [CI] | share <= 0 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in P["best_ensemble_vs_median"]:
        L.append(f"| {MODEL_LABEL[r['model']]} | {r['attack_label']} | "
                 f"{r['best_label']} | {f4(r['ens_mAP'])} | "
                 f"{f4(r['median_mAP'])} | {fd(r['delta'])} | "
                 f"{r['delta']['share_le0']:.3f} |")
    L.append("")
    if P["best_ensemble_vs_median"]:
        names = list(P["best_ensemble_vs_median"][0]["all_ensembles"])
        L.append("Every ensemble - Median (mAP [CI]), so the choice of the best "
                 "one does not drive the verdict:\n")
        L.append("| Model | Attack | " + " | ".join(dlabel(n) for n in names)
                 + " |")
        L.append("|---|---|" + "---|" * len(names))
        for r in P["best_ensemble_vs_median"]:
            L.append(f"| {MODEL_LABEL[r['model']]} | {r['attack_label']} | "
                     + " | ".join(fd(r["all_ensembles"].get(n)) for n in names)
                     + " |")
        L.append("")

    L.append(f"## Score-free detection quality (IoU >= {cfg['sf_iou']:g})\n")
    L.append("One-to-one greedy matching per image and class, pairs ordered "
             "by IoU (score breaks ties only), crowd regions ignored, counts "
             "pooled over images and classes. Cells: precision / recall / F1; "
             "the F1 CI is the paired bootstrap of the same B resamples.\n")
    if "yolo" in cfg["models"]:
        L.append(f"For YOLO read the conf >= {cfg['yolo_sf_conf']:g} table "
                 "first: at the dump threshold (0.001) most boxes are "
                 "low-confidence ones that mAP ranks last but a count metric "
                 "scores as false positives, so ensembles (which merge three "
                 "branches' boxes) look worse there by construction.\n")
    for blk in P["score_free_tables"]:
        L.append(f"### {MODEL_LABEL[blk['model']]}, {blk['label']}\n")
        heads = ["Defense", "Clean P / R / F1"] + [
            f"{a} P / R / F1" for a in blk["attack_labels"]]
        L.append("| " + " | ".join(heads) + " |")
        L.append("|" + "---|" * len(heads))
        for row in blk["rows"]:
            cells = [row["label"]] + [
                ("n/a" if c is None else
                 f"{c['precision']:.3f} / {c['recall']:.3f} / {c['f1']:.3f}")
                for c in row["cells"]]
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
        heads = ["Defense", "Clean cost, F1 [CI]"] + [
            f"{a} recovery, F1 [CI]" for a in blk["attack_labels"]]
        L.append("| " + " | ".join(heads) + " |")
        L.append("|" + "---|" * len(heads))
        for row in blk["rows"][1:]:
            cells = [row["label"]] + [fd(d, 3) for d in row["deltas"]]
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

    bs = P["bootstrap_sanity"]
    L.append("## Bootstrap sanity\n")
    L.append(f"- Bootstrap mean vs point estimate (mAP), over "
             f"{bs['n_conditions']} conditions: median |bias| "
             f"{bs['median_abs_bias']:.4f}, max |bias| "
             f"{bs['max_abs_bias']:.4f} ({bs['max_bias_over_se']:.2f} SE, "
             f"`{bs['max_bias_condition']}`). Typical SE "
             f"{bs['median_se']:.4f}.")
    nf = len(bs["flagged"])
    L.append(f"- Flagged (|bias| > {cfg['bias_flag']:g} SE): "
             + (f"all {nf} conditions" if nf == bs["n_conditions"] and nf > 10
                else f"{nf}: " + ", ".join(f"`{c}`" for c in bs["flagged"])
                if nf else "none") + ".")
    bd = bs.get("decomposition")
    if bd:
        txt = (f"- Where it comes from (measured, medians over "
               f"{bd['n_conditions']} conditions): bias "
               f"{bd['median_total']:+.4f} = class composition "
               f"{bd['median_composition']:+.4f} (classes absent from a "
               f"resample leaving the class mean; max |.| "
               f"{bd['max_abs_composition']:.4f}) + within-class "
               f"{bd['median_within_class']:+.4f} (each class's own AP "
               "drifts up under resampling).")
        if "median_envelope_fraction" in bd:
            txt += (f" Without accumulate()'s running-maximum precision "
                    f"envelope the within-class part falls from "
                    f"{bd['median_within_class_diag']:+.4f} to "
                    f"{bd['median_within_class_no_envelope_diag']:+.4f} "
                    f"(first {bd['n_diag']} resamples), so the envelope is "
                    f"about {bd['median_envelope_fraction']:.0%} of it and "
                    "the rest is AP's own nonlinearity (precision is a ratio "
                    "of resampled counts). A small upward drift per class "
                    "adds up over 80 classes while the SE of their mean "
                    "shrinks, hence about one SE.")
        L.append(txt)
    L.append("- Both effects act on the two terms of a paired difference "
             "alike, so read the difference CIs in the tables above, not CIs "
             "of single mAPs.")
    for mt, dd in (bs.get("deltas") or {}).items():
        L.append(f"- Paired {mt} differences ({dd['n']}): median |bias| "
                 f"{dd['median_abs_bias_over_se']:.2f} SE, max "
                 f"{dd['max_abs_bias_over_se']:.2f} SE (`{dd['max_condition']}`"
                 f", {dd['max_abs_bias']:.4f}). \"Excludes 0\" verdicts "
                 f"that change under a bias-corrected (BC) percentile CI: "
                 f"**{dd['n_verdict_flips_under_bc']}**"
                 + (" (" + ", ".join(f"`{c}`" for c in dd["verdict_flips"])
                    + ")" if dd["verdict_flips"] else "") + ". Such cells "
                 "carry the BC interval inline.")
    L.append("")
    return "\n".join(L)


# ======================================================================
# 9. Main
# ======================================================================
def _git():
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=REPO_ROOT)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5,
                               cwd=REPO_ROOT)
        return (out.stdout.strip() or "unknown") + (
            "-dirty" if dirty.stdout.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def round_floats(obj, sig=7):
    """Floats to `sig` significant digits (JSON size; 4 decimals are cited)."""
    if isinstance(obj, float):
        return float(f"{obj:.{sig}g}") if np.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: round_floats(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, sig) for v in obj]
    return obj


def write_outputs(paths, payload, markdown):
    blob = json.dumps(round_floats(payload), indent=1)
    for key in ("json", "json_latest"):
        with open(paths[key], "w") as f:
            f.write(blob)
    for key in ("md", "md_latest"):
        with open(paths[key], "w") as f:
            f.write(markdown)
    for key in ("json", "json_latest", "md", "md_latest"):
        print(f"[out] {paths[key]}")


def _rel(path):
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return path


def run(args, paths):
    t_start = time.time()
    try:
        import pycocotools  # noqa: F401
    except ImportError:
        raise SystemExit("pycocotools missing: run with "
                         "/home/king/miniconda3/envs/vlm_ftune/bin/python")
    ann = resolve_ann_file(args.ann_file)
    print(f"[cfg] B={args.B} seed={args.seed} ci={args.ci:g}% "
          f"engine={args.engine} workers={args.workers} "
          f"verify_resamples={args.verify_resamples} n={args.n}")
    print(f"[cfg] input root {_rel(args.input_root)}  GT {_rel(ann)}")
    if args.partial:
        print("[cfg] PARTIAL run (subset): outputs are not citable.")

    runs = {(m, a): discover(args.input_root, m, a)
            for m in args.models for a in args.attacks}
    tagmap, tasks, task_meta = {}, {}, {}
    for (m, a), r in runs.items():
        for tag, path in r["files"].items():
            if args.only and tag not in args.only:
                continue
            key = f"{m}/{file_md5(path)}"
            tagmap[(m, a, tag)] = key
            if key not in tasks:
                tasks[key] = {"key": key, "model": m, "path": path}
                task_meta[key] = {"model": m, "tag": tag,
                                  "file": _rel(path), "dirs": []}
            task_meta[key]["dirs"].append(a)
    if not tasks:
        raise SystemExit("no condition selected (check --only)")

    gt, ids, n_all = load_gt(ann, args.n)
    N = len(ids)
    print(f"[gt] {N} of {n_all} val2017 ids (first sorted), "
          f"{len(gt.dataset['annotations'])} GT boxes, digest "
          f"{ids_digest(ids)}")
    for (m, a), r in runs.items():
        n_sum = r["summary"].get("num_images")
        if n_sum != N:
            print(f"[WARN] {m}/{a}: summary.json num_images={n_sum} != {N}")

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, N, size=(args.B, N), dtype=np.int64)
    _CTX.clear()
    _CTX.update(gt=gt, ids=ids, idset=set(ids), idx=idx,
                gt_sf=gt_by_img_cat(gt.dataset["annotations"]),
                sf_iou=args.sf_iou, yolo_sf_conf=args.yolo_sf_conf,
                engine=args.engine, verify_resamples=args.verify_resamples,
                diag_resamples=args.diag_resamples)

    order = sorted(tasks.values(),
                   key=lambda t: -os.path.getsize(t["path"]))
    print(f"[run] {len(order)} unique dumps ({len(tagmap)} condition slots), "
          f"{args.workers} worker(s)")
    results = {}

    def _done(res, k):
        meta = task_meta[res["key"]]
        tm = res["timing"]
        print(f"[eval {k:>2}/{len(order)}] {meta['model']:<8} "
              f"{meta['tag']:<34} dets={res['n_dets']:>6} "
              f"mAP={res['point']['mAP']:.4f} AP50={res['point']['AP50']:.4f}"
              f"  evaluate {tm['evaluate_s']:.0f}s  boot {tm['boot_s']:.0f}s"
              f"  diag {tm['diag_s']:.0f}s")
        results[res["key"]] = res

    if args.workers == 1:
        for k, t in enumerate(order, 1):
            _done(evaluate_condition(t, _CTX), k)
    else:
        # Long-lived workers, all forked here by the main thread before the
        # pool's handler threads exist. maxtasksperchild would re-fork from
        # the handler thread while the main thread prints through run_log's
        # tee; the child inherits a held stdout/run.log lock and hangs at
        # exit, so its slot is lost (reproduced 3/3, 24 Oct). Reusing a
        # worker is safe: --workers 1 already runs every condition in one
        # process.
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            it = pool.imap_unordered(_worker, order)
            for k in range(1, len(order) + 1):
                try:  # a dead or stuck worker fails loudly, never hangs
                    res = it.next(timeout=args.task_timeout)
                except mp.TimeoutError:
                    pool.terminate()
                    raise SystemExit(
                        f"[error] no condition finished within "
                        f"{args.task_timeout:g} s ({k - 1}/{len(order)} done)"
                        "; rerun with --workers 1 to see which one stalls.")
                _done(res, k)

    # Image-id check: the dumps, not the GT, decide whether the "first n sorted
    # ids" rule is right. Every detection must fall inside the scored ids.
    idset = set(ids)
    image_id_check = {}
    for m in args.models:
        rs = [r for r in results.values() if r["model"] == m]
        if not rs:
            continue
        union = set().union(*(r["image_ids_present"] for r in rs))
        outside = sum(r["n_dets_outside_ids"] for r in rs)
        image_id_check[m] = {
            "n_dumps": len(rs), "n_ids_in_dumps": len(union),
            "subset_of_scored_ids": union <= idset,
            "equals_scored_ids": union == idset,
            "n_dets_outside_scored_ids": int(outside),
            "max_id_in_dumps": max(union) if union else None}
        print(f"[ids] {m}: {len(union)} distinct image ids across "
              f"{len(rs)} dumps; subset of the {N} scored ids: "
              f"{union <= idset}; equal: {union == idset}; dets outside: "
              f"{outside}")
        if outside:
            print(f"[WARN] {m}: {outside} detections lie outside the scored "
                  f"ids (dropped). The image rule (--n) may be wrong.")
    for res in results.values():
        res.pop("image_ids_present", None)

    # Score-free bootstrap (multiplicity-weighted counts).
    W = multiplicity_matrix(idx, N)
    for res in results.values():
        res["sf_boot"] = {key: sf_boot(c, W)
                          for key, c in res["score_free_counts"].items()}

    ci = args.ci

    def R(m, a, tag):
        k = tagmap.get((m, a, tag))
        return results.get(k) if k else None

    def D(ra, rb, metric):
        if ra is None or rb is None:
            return None
        return paired_delta(ra["boot"][metric], rb["boot"][metric],
                            ra["point"][metric], rb["point"][metric], ci)

    def DF1(ra, rb, key):
        if ra is None or rb is None:
            return None
        fa = prf(*(ra["score_free_counts"][key][x].sum()
                   for x in ("tp", "fp", "fn")))[2]
        fb = prf(*(rb["score_free_counts"][key][x].sum()
                   for x in ("tp", "fp", "fn")))[2]
        return paired_delta(ra["sf_boot"][key]["f1"], rb["sf_boot"][key]["f1"],
                            float(fa), float(fb), ci)

    # (a) agreement with summary.json
    # Every mAP-valued field of summary.json: the direct ones, plus the two
    # derived differences (attack_damage = clean - attacked, recovery =
    # defended - attacked). recovery_pct is a ratio of those and not checked.
    checks = []
    for (m, a), r in runs.items():
        s, atag = r["summary"], r["attack_tag"]

        def pt(tag, metric):
            res = R(m, a, tag)  # noqa: B023 (called inside this iteration)
            return None if res is None else res["point"][metric]

        def sub(x, y):
            return None if x is None or y is None else x - y

        cm, am = pt("clean", "mAP"), pt(atag, "mAP")
        want = [("clean_mAP", "clean", cm, s.get("clean_mAP")),
                ("clean_AP50", "clean", pt("clean", "AP50"),
                 s.get("clean_AP50")),
                ("attacked_mAP", atag, am, s.get("attacked_mAP")),
                ("attack_damage", f"clean - {atag}", sub(cm, am),
                 s.get("attack_damage"))]
        for dname, v in s["defenses"].items():
            dtag = f"{atag}+{dname}"
            dm = pt(dtag, "mAP")
            want += [(f"defenses.{dname}.mAP", dtag, dm, v.get("mAP")),
                     (f"defenses.{dname}.AP50", dtag, pt(dtag, "AP50"),
                      v.get("AP50")),
                     (f"defenses.{dname}.recovery", f"{dtag} - {atag}",
                      sub(dm, am), v.get("recovery"))]
        for field, tag, got, committed in want:
            if got is None or committed is None:
                continue
            diff = abs(got - committed)
            checks.append({"model": m, "attack": a, "field": field,
                           "tag": tag, "committed": committed,
                           "recomputed": got, "abs_diff": diff,
                           "ok": diff <= args.tolerance})
    mism = [c for c in checks if not c["ok"]]
    agreement = {"tolerance": args.tolerance, "n_checked": len(checks),
                 "n_mismatch": len(mism),
                 "max_abs_diff": max((c["abs_diff"] for c in checks),
                                     default=0.0),
                 "mismatches": mism, "checks": checks}
    print(f"[check] summary.json: {len(checks) - len(mism)}/{len(checks)} "
          f"values within {args.tolerance:g} (max |diff| "
          f"{agreement['max_abs_diff']:.2e})")
    for c in mism:
        print(f"[MISMATCH] {c['model']}/{c['attack']} {c['field']}: committed "
              f"{c['committed']:.6f} recomputed {c['recomputed']:.6f}")

    engine_check = {
        "max_abs_diff_point": max(r["engine_point_diff"]
                                  for r in results.values()),
        "max_abs_diff_resamples": max(
            (d for r in results.values() for d in r["engine_resample_diffs"]),
            default=0.0),
        "n_resamples_checked": sum(len(r["engine_resample_diffs"])
                                   for r in results.values())}
    print(f"[check] fast accumulate vs pycocotools: point max |diff| "
          f"{engine_check['max_abs_diff_point']:.1e}, resamples max |diff| "
          f"{engine_check['max_abs_diff_resamples']:.1e} "
          f"({engine_check['n_resamples_checked']} checked)")

    clean_identical = {}
    for m in args.models:
        sets = [tuple(sorted((t, k) for (mm, aa, t), k in tagmap.items()
                             if mm == m and aa == a and
                             (t == "clean" or t.startswith("clean+"))))
                for a in args.attacks]
        clean_identical[m] = all(s == sets[0] for s in sets)
        if not clean_identical[m]:
            print(f"[WARN] {m}: clean dumps differ across attack dirs; the "
                  f"clean-cost table uses {args.attacks[0]}'s.")

    # (b) clean cost
    clean_cost, clean_point = {}, {}
    for m in args.models:
        a0 = args.attacks[0]
        base = R(m, a0, "clean")
        if base is None:
            continue
        clean_point[m] = base["point"]
        rows = []
        for dname, kind in runs[(m, a0)]["defenses"]:
            r = R(m, a0, f"clean+{dname}")
            if r is None:
                continue
            rows.append({"defense": dname, "label": dlabel(dname),
                         "kind": kind, "mAP": r["point"]["mAP"],
                         "AP50": r["point"]["AP50"],
                         "delta": {mt: D(r, base, mt) for mt in METRICS}})
        clean_cost[m] = rows

    # (c) recovery, damage, best ensemble vs Median
    recovery, best_vs_median = {}, []
    for m in args.models:
        recovery[m] = {}
        for a in args.attacks:
            r_run = runs[(m, a)]
            atag = r_run["attack_tag"]
            att, cln = R(m, a, atag), R(m, a, "clean")
            if att is None:
                continue
            s = r_run["summary"]
            alabel = ATTACK_LABEL[a] + (f" (eps {s['epsilon']:g})"
                                        if s.get("epsilon") else "")
            rows = []
            for dname, kind in r_run["defenses"]:
                r = R(m, a, f"{atag}+{dname}")
                if r is None:
                    continue
                rows.append({"defense": dname, "label": dlabel(dname),
                             "kind": kind, "mAP": r["point"]["mAP"],
                             "AP50": r["point"]["AP50"],
                             "delta": {mt: D(r, att, mt) for mt in METRICS},
                             "vs_clean": D(r, cln, "mAP")})
            recovery[m][a] = {
                "attack_tag": atag, "attack_label": alabel,
                "attacked_mAP": att["point"]["mAP"],
                "attacked_AP50": att["point"]["AP50"],
                "attack_damage": {mt: D(att, cln, mt) for mt in METRICS},
                "defenses": rows}
            med = R(m, a, f"{atag}+{MEDIAN_KEY}")
            ens = [(n, R(m, a, f"{atag}+{n}")) for n, k in r_run["defenses"]
                   if k.startswith("ens")]
            ens = [(n, r) for n, r in ens if r is not None]
            if med is not None and ens:
                best, rb = max(ens, key=lambda x: x[1]["point"]["mAP"])
                best_vs_median.append({
                    "model": m, "attack": a, "attack_label": alabel,
                    "best_ensemble": best, "best_label": dlabel(best),
                    "ens_mAP": rb["point"]["mAP"],
                    "median_mAP": med["point"]["mAP"],
                    "delta": D(rb, med, "mAP"),
                    "delta_AP50": D(rb, med, "AP50"),
                    "all_ensembles": {n: D(r, med, "mAP") for n, r in ens}})

    # Headline check: in how many of these detection settings is each
    # defense's recovery positive (point) / CI-separated from 0?
    headline, hl_index = [], {}
    for m, by_attack in recovery.items():
        for a, blk in by_attack.items():
            for r in blk["defenses"]:
                if r["defense"] not in hl_index:
                    hl_index[r["defense"]] = len(headline)
                    headline.append({"defense": r["defense"],
                                     "label": r["label"], "kind": r["kind"],
                                     "settings": 0, "point_pos": 0,
                                     "ci_above_0": 0, "ci_below_0": 0,
                                     "point_le0_settings": []})
                h = headline[hl_index[r["defense"]]]
                d = r["delta"]["mAP"]
                h["settings"] += 1
                h["point_pos"] += int(d["point"] > 0)
                h["ci_above_0"] += int(d["lo"] > 0)
                h["ci_below_0"] += int(d["hi"] < 0)
                if d["point"] <= 0:
                    h["point_le0_settings"].append(
                        f"{MODEL_LABEL[m]} {ATTACK_LABEL[a]}")

    # (d) score-free tables
    sf_tables = []
    for m in args.models:
        a_list = [a for a in args.attacks if (m, a) in runs]
        defs = runs[(m, a_list[0])]["defenses"]
        for key, _thr in sf_thresholds(m, args.yolo_sf_conf):
            rows = []
            for dname in [None] + [d for d, _ in defs]:
                ctag = "clean" if dname is None else f"clean+{dname}"
                cells, deltas = [], []
                rc = R(m, a_list[0], ctag)
                cbase = R(m, a_list[0], "clean")
                for col, a in [(None, a_list[0])] + [(a, a) for a in a_list]:
                    atag = runs[(m, a)]["attack_tag"]
                    tag = ctag if col is None else (
                        atag if dname is None else f"{atag}+{dname}")
                    r = R(m, a, tag)
                    if r is None:
                        cells.append(None)
                    else:
                        c = r["score_free_counts"][key]
                        tp, fp, fn = (int(c[x].sum())
                                      for x in ("tp", "fp", "fn"))
                        p, rr, f = prf(tp, fp, fn)
                        lo, hi = percentile_ci(r["sf_boot"][key]["f1"], ci)
                        cells.append({"tp": tp, "fp": fp, "fn": fn,
                                      "ignored": int(c["ignored"].sum()),
                                      "precision": float(p),
                                      "recall": float(rr), "f1": float(f),
                                      "f1_lo": lo, "f1_hi": hi})
                    if dname is not None:
                        if col is None:
                            deltas.append(DF1(rc, cbase, key))
                        else:
                            deltas.append(DF1(r, R(m, a, atag), key))
                rows.append({"defense": dname or "none",
                             "label": "none" if dname is None
                             else dlabel(dname),
                             "cells": cells, "deltas": deltas})
            sf_tables.append({
                "model": m, "key": key, "label": sf_label(key, m),
                "attack_labels": [ATTACK_LABEL[a] for a in a_list],
                "rows": rows})

    # Per-condition record + bootstrap sanity
    conditions, sanity = [], []
    for key, res in results.items():
        meta = task_meta[key]
        bsum = {mt: boot_summary(res["point"][mt], res["boot"][mt], ci)
                for mt in METRICS}
        name = f"{meta['model']}/{meta['tag']}"
        sanity.append((name, bsum["mAP"]))
        conditions.append({
            "model": meta["model"], "tag": meta["tag"], "file": meta["file"],
            "attack_dirs": sorted(set(meta["dirs"])),
            "md5": key.split("/", 1)[1], "n_dets": res["n_dets"],
            "n_dets_outside_ids": res["n_dets_outside_ids"],
            "mAP": res["point"]["mAP"], "AP50": res["point"]["AP50"],
            "coco_stats12": res["stats"], "bootstrap": bsum,
            "bias_decomposition": res["bias_decomposition"],
            "timing_s": res["timing"]})
    conditions.sort(key=lambda c: (c["model"], c["tag"]))
    abs_bias = [abs(b["bias"]) for _, b in sanity]
    worst = max(sanity, key=lambda x: abs(x[1]["bias"]))
    flagged = [n for n, b in sanity
               if abs(b["bias_over_se"]) > args.bias_flag]
    bootstrap_sanity = {
        "n_conditions": len(sanity),
        "median_abs_bias": float(np.median(abs_bias)),
        "max_abs_bias": float(abs(worst[1]["bias"])),
        "max_bias_over_se": float(abs(worst[1]["bias_over_se"])),
        "max_bias_condition": worst[0],
        "median_se": float(np.median([b["se"] for _, b in sanity])),
        "flagged": sorted(flagged), "flag_threshold_se": args.bias_flag}
    print(f"[check] bootstrap mean vs point (mAP): median |bias| "
          f"{bootstrap_sanity['median_abs_bias']:.4f}, max "
          f"{bootstrap_sanity['max_abs_bias']:.4f} "
          f"({bootstrap_sanity['max_bias_over_se']:.2f} SE, {worst[0]}); "
          f"flagged {len(flagged)}")
    decs = [c["bias_decomposition"] for c in conditions
            if c["bias_decomposition"]]
    if decs:
        med = lambda k: float(np.median([d[k] for d in decs]))  # noqa: E731
        bd = {"n_conditions": len(decs), "median_total": med("total"),
              "median_composition": med("composition"),
              "max_abs_composition": float(max(abs(d["composition"])
                                               for d in decs)),
              "median_within_class": med("within_class")}
        dd = [d for d in decs if d["n_diag"] and d["within_class_diag"] != 0]
        if dd:
            bd.update(
                n_diag=int(min(d["n_diag"] for d in dd)),
                median_within_class_diag=float(np.median(
                    [d["within_class_diag"] for d in dd])),
                median_within_class_no_envelope_diag=float(np.median(
                    [d["within_class_no_envelope_diag"] for d in dd])),
                median_envelope_share_diag=float(np.median(
                    [d["envelope_share_diag"] for d in dd])),
                median_envelope_fraction=float(np.median(
                    [d["envelope_share_diag"] / d["within_class_diag"]
                     for d in dd])))
        bootstrap_sanity["decomposition"] = bd
        print(f"[check] bias decomposition (medians over {len(decs)}): total "
              f"{bd['median_total']:+.4f} = composition "
              f"{bd['median_composition']:+.4f} + within-class "
              f"{bd['median_within_class']:+.4f}"
              + (f"; envelope share of within-class "
                 f"{bd['median_envelope_fraction']:.0%} (first "
                 f"{bd['n_diag']} resamples)" if dd else ""))

    # The reported CIs are on paired DIFFERENCES; their bias is what matters.
    bootstrap_sanity["deltas"] = {}
    for mt in METRICS:
        deltas = []
        for m, rows in clean_cost.items():
            deltas += [(f"{m}/clean cost/{r['defense']}", r["delta"][mt])
                       for r in rows]
        for m, by_attack in recovery.items():
            for a, blk in by_attack.items():
                deltas.append((f"{m}/{a}/attack damage",
                               blk["attack_damage"][mt]))
                deltas += [(f"{m}/{a}/recovery/{r['defense']}",
                            r["delta"][mt]) for r in blk["defenses"]]
        for r in best_vs_median:
            if mt == "mAP":
                deltas += [(f"{r['model']}/{r['attack']}/{n} - median", d)
                           for n, d in r["all_ensembles"].items()]
            else:
                deltas.append((f"{r['model']}/{r['attack']}/"
                               f"{r['best_ensemble']} - median",
                               r["delta_AP50"]))
        deltas = [(n, d) for n, d in deltas if d is not None]
        if not deltas:
            continue
        rb = [abs(d["bias_over_se"]) for _, d in deltas]
        wn, wd = max(deltas, key=lambda x: abs(x[1]["bias_over_se"]))
        flips = sorted(n for n, d in deltas
                       if d["excludes_0"] != d["bc_excludes_0"])
        bootstrap_sanity["deltas"][mt] = {
            "n": len(deltas), "median_abs_bias_over_se": float(np.median(rb)),
            "max_abs_bias_over_se": float(max(rb)), "max_condition": wn,
            "max_abs_bias": float(abs(wd["bias"])),
            "n_verdict_flips_under_bc": len(flips), "verdict_flips": flips}
        print(f"[check] paired {mt} differences ({len(deltas)}): median "
              f"|bias| {np.median(rb):.2f} SE, max {max(rb):.2f} SE ({wn}); "
              f"'excludes 0' verdict changes under the bias-corrected CI: "
              f"{len(flips)}" + (f" ({', '.join(flips)})" if flips else ""))

    payload = {
        "experiment": EXPERIMENT, "run_id": args.run_id,
        "partial": args.partial, "git": _git(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"B": args.B, "seed": args.seed, "ci": args.ci,
                   "engine": args.engine,
                   "verify_resamples": args.verify_resamples,
                   "workers": args.workers, "n": args.n,
                   "models": args.models, "attacks": args.attacks,
                   "only": args.only, "tolerance": args.tolerance,
                   "sf_iou": args.sf_iou, "yolo_sf_conf": args.yolo_sf_conf,
                   "bias_flag": args.bias_flag,
                   "diag_resamples": args.diag_resamples,
                   "task_timeout_s": args.task_timeout},
        "image_ids": {"n": N, "digest": ids_digest(ids), "first": ids[0],
                      "last": ids[-1],
                      "rule": "first n of sorted(val2017 GT image ids), "
                              "= Phase 3's sorted(os.listdir(val2017))[:n]"},
        "inputs": {f"{m}/{a}": {"dir": _rel(r["dir"]),
                                "attack_tag": r["attack_tag"],
                                "summary_num_images":
                                    r["summary"].get("num_images")}
                   for (m, a), r in runs.items()},
        "clean_dumps_identical": clean_identical,
        "image_id_check": image_id_check,
        "agreement": agreement, "engine_check": engine_check,
        "clean_point": clean_point, "clean_cost": clean_cost,
        "recovery": recovery, "best_ensemble_vs_median": best_vs_median,
        "headline_check": headline,
        "score_free_tables": sf_tables,
        "bootstrap_sanity": bootstrap_sanity, "conditions": conditions,
        "runtime_s": round(time.time() - t_start, 1),
    }
    write_outputs(paths, payload, build_markdown(payload))
    print(f"[done] {len(results)} dumps, {payload['runtime_s']:.0f} s")
    return 2 if mism else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.list:
        for m in args.models:
            for a in args.attacks:
                r = discover(args.input_root, m, a)
                print(f"{m}/{a}: {_rel(r['dir'])}  attack_tag="
                      f"{r['attack_tag']}")
                for t, p in r["files"].items():
                    print(f"    {t:<36} {os.path.basename(p)}")
        return 0
    os.makedirs(args.output_dir, exist_ok=True)
    paths = output_paths(args.output_dir, args.run_id)
    for k, pth in paths.items():
        assert is_committed(os.path.basename(pth)), pth
    with run_log(paths["log"], argv=["phase4/p4_rescore.py"] + argv,
                 extra={"run_id": args.run_id}):
        code = run(args, paths)
        artifact_report(args.output_dir)
    if code:
        print(f"[exit {code}] recomputed values disagree with summary.json "
              f"(see MISMATCH lines); outputs were still written.")
    return code


if __name__ == "__main__":
    sys.exit(main())
