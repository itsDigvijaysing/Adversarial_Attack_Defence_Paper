#!/usr/bin/env python3
# ======================================================================
# p4_core.py — shared core for every Phase-4 run.
#
# Phase 4 has exactly TWO entry points, one per model:
#
#     python phase4/p4_yolo.py     --experiment {clean_control,adaptive_bpda}
#     python phase4/p4_florence.py --experiment {clean_control}
#
# Everything that is not model-specific lives HERE and is written once: the CLI,
# image selection, the defense branches, the ensemble merge, the checkpointed
# per-image loop, COCO/OCR scoring, the report writers, the run-artifact policy
# and the timing estimates. The two model files contain only what genuinely
# differs — how the model loads, how it infers, and how it is attacked.
#
# WHY THIS SHAPE (Phase-3 lesson): Phase 3 shipped nine near-identical scripts,
# each carrying its own copy of the inference, scoring and reporting code. Every
# fix had to be made nine times, and the copies drifted (different PGD alpha,
# different clamp bounds, different dataset paths). Phase 4 keeps the per-model
# split for EXECUTION — one process, one model, one GPU, one tmux pane — but not
# for logic.
#
# CODE-PATH IDENTITY WITH PHASE 3
# The defense transforms and the class-aware NMS merge are imported from the
# repo-root `phase3_common.py` (`apply_all_defenses_gpu`, `svd_gpu`,
# `bilateral_gpu`, `merge_branches_nms`, `assemble_results`, `SurveyCheckpoint`,
# `config_signature`). They are never reimplemented here, which is the only
# reason Phase-4 numbers may be compared against the Phase-3 attacked runs.
# `phase3_common.py` therefore stays at the repo root, NOT in archive/Phase3/.
#
# OUTPUT LAYOUT
#   results/phase4/<experiment>_<model>/
#       summary_<experiment>_<model>[_<track>]_n<N>_<timestamp>.json   immutable
#       summary_..._latest.json / .md      overwritten mirror, stable to cite
#       run.log                            appended, one stamped block per run
#       checkpoint_<track>_<sig8>.pkl      config-keyed, so resume still works
#   Timestamped summaries never overwrite each other; the directory is stable so
#   a crashed run resumes instead of restarting. The per-condition COCO dumps go
#   to a temp dir and are deleted (see p4_logging.py).
# ======================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher

PHASE4_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PHASE4_DIR)
for _p in (PHASE4_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from p4_logging import artifact_report, eval_dump_dir, run_log  # noqa: E402

DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "phase4")

# ----------------------------------------------------------------------
# Defense naming. Branch keys are phase3_common's; the labels are the paper's.
# ----------------------------------------------------------------------
SOLO_PAPER_NAMES = {
    "jpeg": "JPEG",
    "median": "Median",
    "gaussian": "Gaussian",
    "tvm": "TVM",
    "blur_tvm": "Blur+TVM",
    "bilateral": "Bilateral",
    # Not one of the six headline solos: produced because EnsJTS needs it as a
    # branch, and reported because the row is then free.
    "svd": "SVD (ensemble member)",
}
ENSEMBLE_PAPER_NAMES = {
    "ens_jpeg_median_gaussian": "EnsJMG",
    "ens_jpeg_median_tvm": "EnsJMT",
    "ens_4way": "Ens4W",
    "ens_blur_tvm_combo": "EnsBTC",
    "ens_bilateral_median": "EnsBMT",
    "ens_jpeg_bilateral_tvm": "EnsJBT",
    "ens_jpeg_tvm_svd": "EnsJTS",
}

# Published N=5000 baselines, used ONLY for the "delta vs published" column.
# Provenance: Florence-2 detection from the committed summaries in
# results/results_survey_florence_detection_*/; YOLO and OCR from the runs done
# on teammates' machines, whose summary files are not in this repo yet.
PUBLISHED_BASELINES = {
    "yolo": 0.4519,
    "florence_det": 0.3300,
    "florence_ocr": 1.000,
}
TRACK_METRIC = {
    "yolo": "COCO mAP@[.5:.95]",
    "florence_det": "COCO mAP@[.5:.95]",
    "florence_ocr": "OCR self-consistency similarity",
}


class _Runtime:
    """Holder for the heavy modules, populated by load_runtime().

    A shared object rather than module globals so p4_yolo / p4_florence see the
    same instance after the GPU has been pinned.
    """

    np = None
    torch = None
    Image = None
    pc = None


rt = _Runtime()


# ======================================================================
# 1. CLI — stdlib only, so --help and --preflight work with no torch.
# ======================================================================
def base_parser(description, experiments):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--experiment", choices=experiments, default=experiments[0],
                   help="Which Phase-4 experiment to run.")
    p.add_argument("--n", "--num-images", dest="n", type=int, default=1000,
                   help="Number of images (default 1000).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--select", choices=["head", "random"], default="head",
                   help="head (default) = first --n of the sorted file list, "
                        "i.e. the subset the Phase-3 attacked runs scored, so "
                        "deltas are same-subset. random = seeded sample.")
    p.add_argument("--image-dir", default=None,
                   help="COCO val images. Default: auto-resolve ./val2017, "
                        "./Dataset/val2017, ...")
    p.add_argument("--ann-file", default=None,
                   help="COCO instances_val2017.json. Default: auto-resolve.")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index; <0 picks the freest via nvidia-smi.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--output-dir", default=None,
                   help="Override the whole output path (default: "
                        "<results-root>/<experiment>_<model>/).")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--keep-eval-dumps", action="store_true",
                   help="Keep the per-condition COCO dumps (hundreds of MB, "
                        "gitignored). Off by default so a run directory stays "
                        "shareable.")
    p.add_argument("--no-per-image", action="store_true",
                   help="Skip the per-image audit JSON (gitignored either way).")
    p.add_argument("--smoke-test", action="store_true",
                   help="2 images on CPU into <output-dir>/smoke_test, stamped "
                        "smoke_test=true. Proves the plumbing; NOT a result.")
    p.add_argument("--preflight", action="store_true",
                   help="Check dependencies / data / weights, then exit.")

    # Defense hyperparameters — defaults MUST match the Phase-3 survey runs.
    g = p.add_argument_group("defenses (Phase-3 values)")
    g.add_argument("--jpeg-quality", type=int, default=75)
    g.add_argument("--median-kernel", type=int, default=3)
    g.add_argument("--tvm-weight", type=float, default=0.05)
    g.add_argument("--tvm-iters", type=int, default=200)
    g.add_argument("--gaussian-sigma", type=float, default=1.0)
    g.add_argument("--svd-keep-ratio", type=float, default=0.90)
    g.add_argument("--nms-iou", type=float, default=0.5,
                   help="Inference NMS and ensemble merge IoU (Phase 3: 0.5).")
    return p


def finalize_args(args, model_key):
    """Resolve the run id, the output dir and the --smoke-test overrides."""
    args.model_key = model_key
    # Timestamp fixed once per process and stamped into every filename it writes.
    args.run_id = time.strftime("%Y%m%d-%H%M%S")
    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.results_root, f"{args.experiment}_{model_key}")
    if args.smoke_test:
        args.n = 2
        args.device = "cpu"
        args.no_checkpoint = True
        args.output_dir = os.path.join(args.output_dir, "smoke_test")
    return args


# ======================================================================
# 2. Paths, preflight, image selection.
# ======================================================================
def resolve_image_dir(primary):
    candidates = [primary] if primary else []
    candidates += [os.path.join(REPO_ROOT, "val2017"),
                   os.path.join(REPO_ROOT, "Dataset", "val2017"),
                   "val2017", "./val2017", "Dataset/val2017",
                   "./Dataset/val2017", "dataset/val2017"]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    raise RuntimeError("Could not find the COCO image directory. Tried: "
                       + ", ".join(str(c) for c in candidates))


def resolve_ann_file(primary):
    candidates = [primary] if primary else []
    candidates += [os.path.join(REPO_ROOT, "annotations",
                                "instances_val2017.json"),
                   os.path.join(REPO_ROOT, "Dataset", "annotations",
                                "instances_val2017.json"),
                   "annotations/instances_val2017.json",
                   "Dataset/annotations/instances_val2017.json"]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise RuntimeError("Could not find instances_val2017.json. Tried: "
                       + ", ".join(str(c) for c in candidates))


def preflight(args, modules, weights):
    """Shared preflight. `modules` = [(name, needed_by)], `weights` = [(label, present_bool_fn)]."""
    import importlib.util
    print("=" * 70)
    print(f"Phase-4 preflight — {args.model_key} / {args.experiment}".center(70))
    print("=" * 70)
    ok = True
    print("\n[python packages]")
    for mod, needed_by in modules:
        found = importlib.util.find_spec(mod) is not None
        if not found:
            ok = False
        print(f"  {mod:<14} {'OK' if found else 'MISSING':<8} ({needed_by})")
    print("\n[data]")
    for label, fn, val in (("images", resolve_image_dir, args.image_dir),
                           ("annotations", resolve_ann_file, args.ann_file)):
        try:
            print(f"  {label:<14} OK       {fn(val)}")
        except RuntimeError as exc:
            ok = False
            print(f"  {label:<14} MISSING  {exc}")
    print("\n[weights]")
    for label, present in weights:
        print(f"  {label:<26} {'present/cached' if present else 'will download'}")
    print("\n[shared core]")
    core = os.path.join(REPO_ROOT, "phase3_common.py")
    if not os.path.isfile(core):
        ok = False
    print(f"  phase3_common.py {'OK' if os.path.isfile(core) else 'MISSING'}"
          f"       {core}")
    print("\n[verdict] " + ("READY" if ok else "NOT READY — see MISSING above"))
    print("=" * 70)
    return 0 if ok else 1


def select_images(image_dir, n, seed, mode):
    # Phase 3 used a bare sorted(os.listdir(...)); the extension filter is a
    # strict improvement that changes nothing on a clean COCO val2017 dir, so
    # `--select head` reproduces exactly the subset the attacked runs scored.
    files = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise RuntimeError(f"No images found in {image_dir}")
    chosen = (files[:n] if mode == "head"
              else sorted(random.Random(seed).sample(files, min(n, len(files)))))
    if len(chosen) < n:
        print(f"[WARN] requested n={n} but only {len(chosen)} images exist; "
              f"treating {len(chosen)} as the requested set.")
    ids = []
    for f in chosen:
        stem = os.path.splitext(f)[0]
        assert stem.isdigit(), (
            f"image '{f}' is not a COCO-style numeric filename; the detection "
            f"tracks need the COCO image_id to score against the GT.")
        ids.append(int(stem))
    assert len(set(ids)) == len(ids), "duplicate image ids in the selection"
    return chosen, ids


def ids_digest(ids) -> str:
    return hashlib.sha1(json.dumps(sorted(ids)).encode()).hexdigest()[:16]


# ======================================================================
# 3. Runtime import (CUDA_VISIBLE_DEVICES must be set before torch).
# ======================================================================
def _pick_freest_gpu() -> str:
    import subprocess
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        gpu_free = []
        for line in smi.stdout.strip().split("\n"):
            idx, free = line.split(",")
            gpu_free.append((int(idx.strip()), int(free.strip())))
        gpu_free.sort(key=lambda x: x[1], reverse=True)
        if gpu_free:
            return str(gpu_free[0][0])
    except Exception as exc:  # noqa: BLE001
        print(f"[GPU] nvidia-smi failed ({exc}); using device 0.")
    return "0"


def load_runtime(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[GPU] CUDA_VISIBLE_DEVICES='' (CPU run)")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            _pick_freest_gpu() if args.gpu < 0 else str(args.gpu))
        print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    try:
        import numpy as _np
        import torch as _torch
        from PIL import Image as _Image
        import phase3_common as _pc
    except ImportError as exc:
        raise SystemExit(
            f"[fatal] missing dependency: {exc}\n"
            f"        Activate the project environment (`conda activate "
            f"vlm_ftune`) and re-run; `--preflight` lists what is missing.\n"
            f"        phase3_common.py must be at the repo root: {REPO_ROOT}"
        ) from exc
    rt.np, rt.torch, rt.Image, rt.pc = _np, _torch, _Image, _pc
    if args.device == "cpu":
        return rt.torch.device("cpu")
    assert rt.torch.cuda.is_available(), (
        "CUDA is not available. Use --device cpu (slow) or --smoke-test.")
    return rt.torch.device("cuda:0")


# ======================================================================
# 4. Defense branches — phase3_common's own transforms, never reimplemented.
# ======================================================================
def load_ensembles(keys=None):
    keys = list(ENSEMBLE_PAPER_NAMES) if keys is None else list(keys)
    out = {}
    for key in keys:
        assert key in rt.pc.SURVEY_ENSEMBLES, (
            f"ensemble '{key}' is not in phase3_common.SURVEY_ENSEMBLES; "
            f"Phase 4 must score the same ensembles Phase 3 did.")
        out[key] = list(rt.pc.SURVEY_ENSEMBLES[key])
    return out


def branch_names(ensembles):
    """The requested solos plus every ensemble member, order-stable."""
    names = list(SOLO_PAPER_NAMES.keys())
    for members in ensembles.values():
        for m in members:
            assert m in names, (
                f"ensemble member '{m}' is not a Phase-4 branch; branches={names}")
    return names


def build_branches(pil_img, device, args, names):
    """{branch: PIL} for the requested branches.

    `pc.apply_all_defenses_gpu` produces the 5 locked solos exactly as Phase 3
    did (one GPU upload, TVM batched over [src, blurred] so the numerics match);
    svd/bilateral are then added with the same two lines
    `pc.apply_survey_defenses` uses (phase3_common.py:411-413). When neither
    tvm nor blur_tvm is requested the primitives are called directly instead,
    which is bit-identical for those branches and skips ~200 wasted TVM
    iterations per call.
    """
    pc = rt.pc
    need_tvm = ("tvm" in names) or ("blur_tvm" in names)
    if need_tvm:
        out = pc.apply_all_defenses_gpu(
            pil_img, device,
            jpeg_quality=args.jpeg_quality, median_kernel=args.median_kernel,
            tvm_weight=args.tvm_weight, tvm_iters=args.tvm_iters,
            gaussian_sigma=args.gaussian_sigma)
    else:
        out = {}
        if "jpeg" in names:
            out["jpeg"] = pc.jpeg_cpu(pil_img, args.jpeg_quality)
        if {"median", "gaussian"} & set(names):
            src0 = pc.pil_to_tensor(pil_img, device)
            if "median" in names:
                out["median"] = pc.tensor_to_pil(
                    pc.median_gpu(src0, args.median_kernel))
            if "gaussian" in names:
                out["gaussian"] = pc.tensor_to_pil(
                    pc.gaussian_gpu(src0, sigma=args.gaussian_sigma))
    if "svd" in names or "bilateral" in names:
        src = pc.pil_to_tensor(pil_img, device)
        if "svd" in names:
            out["svd"] = pc.tensor_to_pil(
                pc.svd_gpu(src, keep_ratio=args.svd_keep_ratio))
        if "bilateral" in names:
            out["bilateral"] = pc.tensor_to_pil(pc.bilateral_gpu(src))
    missing = [n for n in names if n not in out]
    assert not missing, f"defense producer did not return {missing}"
    return {n: out[n] for n in names}


# ======================================================================
# 5. OCR metric helpers — VERBATIM from the Phase-3 OCR scripts
#    (archive/Phase3/run_survey_florence_ocr.py:151-208).
#    The metric is SELF-CONSISTENCY, not accuracy: char-level SequenceMatcher
#    against the model's OWN undefended output. Never call it word recovery.
# ======================================================================
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def decode_ocr(parsed_or_text) -> str:
    if isinstance(parsed_or_text, dict):
        val = parsed_or_text.get("<OCR>", "")
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            return " ".join(str(x) for x in val)
        if isinstance(val, dict):
            if "text" in val:
                return str(val["text"])
            return json.dumps(val, ensure_ascii=True)
    return str(parsed_or_text)


def sanitize_generated_text(raw: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", cleaned).strip()


def consensus_vote_text(texts):
    """The OCR analogue of the detection NMS merge (no boxes to merge)."""
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    norm = [normalize_text(t) for t in texts]
    counts = Counter(norm)
    top_norm, top_count = counts.most_common(1)[0]
    tied = [k for k, v in counts.items() if v == top_count]
    if len(tied) == 1 and top_count >= 2:
        for t in texts:
            if normalize_text(t) == top_norm:
                return t
    pairwise = []
    for i, ti in enumerate(texts):
        sim_sum = 0.0
        for j, tj in enumerate(texts):
            if i == j:
                continue
            sim_sum += text_similarity(ti, tj)
        pairwise.append(sim_sum)
    best_idx = max(range(len(texts)),
                   key=lambda i: (pairwise[i], len(norm[i]), len(texts[i])))
    return texts[best_idx]


# ======================================================================
# 6. Checkpoint + the generic checkpointed per-image loop.
# ======================================================================
def make_checkpoint(args, track, sig):
    """Config-keyed checkpoint whose FILENAME also carries the signature.

    Phase 3 used one fixed filename per attack and relied on the signature
    stored inside it, so changing a hyperparameter invalidated the previous
    cache and threw the work away. Putting sig8 in the name lets several
    configurations coexist in one (stable, resumable) directory.
    """
    path = os.path.join(args.output_dir, f"checkpoint_{track}_{sig[:8]}.pkl")
    return rt.pc.SurveyCheckpoint(path, sig, flush_every=args.checkpoint_every,
                                  enabled=not args.no_checkpoint)


def run_per_image(args, label, files, conditions, compute_one, ckpt,
                  log_every=25):
    """Run `compute_one(pil, img_id) -> {tag: value}` over `files`.

    Returns (elapsed_seconds, computed_count, failures). `computed` counts only
    images actually processed this run — checkpoint-resumed images cost nothing,
    so dividing by len(files) would report a fake speed and wreck the runtime
    estimate. Asserts, after the loop, that every image carries every condition.
    """
    start = time.time()
    failures, computed = [], 0
    for i, fname in enumerate(files):
        img_id = int(os.path.splitext(fname)[0])
        if ckpt.has(img_id):
            continue
        try:
            pil_img = rt.Image.open(
                os.path.join(args.image_dir, fname)).convert("RGB")
            ckpt.put(img_id, compute_one(pil_img, img_id))
            computed += 1
        except Exception as exc:  # noqa: BLE001 — one bad image never kills a run
            failures.append((fname, repr(exc)))
        if (i + 1) % log_every == 0 or (i + 1) == len(files):
            el = (time.time() - start) / 60.0
            rate = (el * 60 / computed) if computed else float("nan")
            print(f"[{label}] {i + 1}/{len(files)} | {el:.1f}m | {rate:.2f}s/img")
        if rt.torch.cuda.is_available() and (i + 1) % 250 == 0:
            rt.torch.cuda.empty_cache()

    elapsed = time.time() - start
    if failures:
        print(f"[{label}] {len(failures)} image(s) failed "
              f"(first: {failures[0]}).")
    assert len(ckpt.buckets) == len(files), (
        f"[{label}] image count mismatch: processed {len(ckpt.buckets)} but "
        f"expected {len(files)} ({len(failures)} failures).")
    for iid, b in ckpt.buckets.items():
        missing = [c for c in conditions if c not in b]
        assert not missing, f"[{label}] img {iid} missing conditions: {missing}"
    print(f"[{label}] all {len(ckpt.buckets)} images carry all "
          f"{len(conditions)} conditions.")
    return elapsed, computed, failures


def record_dets(buckets, tag, dets, img_id):
    """Stamp image_id onto a detection list and file it under `tag`."""
    for d in dets:
        d["image_id"] = img_id
    buckets[tag] = dets
    return dets


# ======================================================================
# 7. Scoring.
# ======================================================================
def evaluate(args, all_results, coco_gt, image_ids):
    """COCO-eval every condition, with the bulky dumps sent to a temp dir."""
    with eval_dump_dir(args.output_dir, keep=args.keep_eval_dumps) as dumps:
        return rt.pc.evaluate_all_conditions(
            all_results, coco_gt, image_ids=sorted(image_ids),
            output_dir=dumps)


def score_detection(args, ckpt, names, ensembles, coco_gt, attack_tags=(),
                    baseline_tag="clean"):
    """Assemble ensembles via the Phase-3 NMS merge, then COCO-eval everything.

    Returns (eval_stats, {condition_tag: {mAP, AP50}}).
    """
    all_results = rt.pc.assemble_results(
        ckpt.buckets, defense_names=names, attack_tags=list(attack_tags),
        ensembles=ensembles, nms_iou=args.nms_iou)
    eval_stats = evaluate(args, all_results, coco_gt, ckpt.buckets.keys())
    conditions = {t: {"mAP": float(eval_stats[t][0]),
                      "AP50": float(eval_stats[t][1])}
                  for t in all_results}
    assert baseline_tag in conditions, f"missing baseline condition {baseline_tag}"
    return eval_stats, conditions


def defense_table(conditions, names, ensembles, prefix, baseline_tag):
    """{defense: {value, AP50, delta_vs_measured, kind}} for one condition prefix."""
    base = conditions[baseline_tag]["mAP"]
    out = {}
    for name in list(names) + list(ensembles):
        tag = f"{prefix}+{name}"
        assert tag in conditions, f"no eval for condition {tag}"
        val = conditions[tag]["mAP"]
        out[name] = {"value": val, "AP50": conditions[tag]["AP50"],
                     "delta_vs_measured": val - base,
                     "kind": "ENSEMBLE" if name in ensembles else "solo"}
    return out


def score_ocr(ckpt, names, ensembles, ref_tag, prefix):
    """Average self-consistency per branch/ensemble, plus the per-image rows."""
    sums = {n: 0.0 for n in list(names) + list(ensembles)}
    per_image = {}
    for iid, texts in ckpt.buckets.items():
        ref = texts[ref_tag]
        row = {}
        for n in names:
            s = text_similarity(ref, texts[f"{prefix}+{n}"])
            sums[n] += s
            row[n] = s
        for ename, members in ensembles.items():
            voted = consensus_vote_text([texts[f"{prefix}+{m}"] for m in members])
            s = text_similarity(ref, voted)
            sums[ename] += s
            row[ename] = s
        per_image[str(iid)] = row
    n_img = max(1, len(ckpt.buckets))
    table = {n: {"value": sums[n] / n_img,
                 "delta_vs_measured": sums[n] / n_img - 1.0,
                 "kind": "ENSEMBLE" if n in ensembles else "solo"}
             for n in sums}
    return table, per_image


# ======================================================================
# 8. Output files. Timestamped (immutable) + `_latest` (stable to cite).
# ======================================================================
def write_per_image(args, name, payload):
    if args.no_per_image:
        return None
    path = os.path.join(args.output_dir, f"per_image_{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def write_summary(args, payload, stem):
    """Write summary_<stem>_n<N>_<run_id>.json plus the _latest mirror."""
    payload = dict(payload)
    payload["run_id"] = args.run_id
    base = f"summary_{stem}_n{args.n}"
    stamped = os.path.join(args.output_dir, f"{base}_{args.run_id}.json")
    latest = os.path.join(args.output_dir, f"summary_{stem}_latest.json")
    blob = json.dumps(payload, indent=2)
    for path in (stamped, latest):
        with open(path, "w") as f:
            f.write(blob)
    print(f"[out] {stamped}\n[out] {latest}  (mirror of the newest run)")
    return stamped, latest


def write_markdown(args, text, stem):
    stamped = os.path.join(args.output_dir,
                           f"summary_{stem}_n{args.n}_{args.run_id}.md")
    latest = os.path.join(args.output_dir, f"summary_{stem}_latest.md")
    for path in (stamped, latest):
        with open(path, "w") as f:
            f.write(text)
    print(f"[out] {stamped}\n[out] {latest}")
    return stamped, latest


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_delta(v, nd=4):
    return "n/a" if v is None else f"{v:+.{nd}f}"


def header_lines(args, image_ids, extra=()):
    lines = []
    if args.smoke_test:
        lines.append("> **SMOKE TEST OUTPUT — NOT A RESULT.** 2 images, CPU. "
                     "Do not cite.\n")
    lines.append(f"- Run `{args.run_id}` · model `{args.model_key}` · "
                 f"experiment `{args.experiment}`")
    lines.append(f"- Images: **n = {len(image_ids)}** (`--select {args.select}`, "
                 f"seed {args.seed}, id digest `{ids_digest(image_ids)}`)")
    lines.append(f"- Image dir `{args.image_dir}` · annotations "
                 f"`{args.ann_file}`")
    lines.extend(extra)
    lines.append("")
    return lines


def timing_lines(elapsed, computed, total, device, targets=(1000, 5000)):
    lines = ["## Timing\n"]
    if not computed:
        lines.append("- every image was resumed from checkpoint; nothing was "
                     "timed this run.\n")
        return lines
    spi = elapsed / computed
    lines.append(f"- **{spi:.2f} s/image** on `{device}` — {elapsed / 60:.1f} min "
                 f"for {computed} of {total} images computed this run "
                 f"(resumed images excluded from the rate)")
    est = " · ".join(f"n={t}: {spi * t / 3600:.2f} h" for t in targets)
    lines.append(f"- extrapolated: {est}\n")
    return lines


def timing_table(results, device, targets=(1000, 5000)):
    """Per-track timing table for a multi-track run (clean_control)."""
    lines = ["## Timing\n",
             "s/image is measured over the images actually computed this run; "
             "checkpoint-resumed images are excluded, so a resumed run still "
             f"reports a real rate. Device: `{device}`.\n"]
    head = "| Track | images | computed now | s/image | this run | " + \
           " | ".join(f"est. n={t}" for t in targets) + " |"
    lines.append(head)
    lines.append("|---|---|---|---|---|" + "---|" * len(targets))
    for track, r in results.items():
        spi = r["seconds_per_image"]
        if spi is None:
            lines.append(f"| {track} | {r['num_images']} | 0 | n/a (all "
                         f"resumed) | {r['seconds'] / 60:.1f} min |"
                         + " n/a |" * len(targets))
            continue
        est = " | ".join(f"{spi * t / 3600:.2f} h" for t in targets)
        lines.append(f"| {track} | {r['num_images']} | "
                     f"{r['images_computed_this_run']} | {spi:.2f} | "
                     f"{r['seconds'] / 60:.1f} min | {est} |")
    lines.append("")
    return lines


def print_timing(results, device, targets=(1000, 5000)):
    print("\n" + "=" * 70)
    print("RUNTIME ESTIMATE (from this run's own s/image)")
    print("=" * 70)
    totals = {t: 0.0 for t in targets}
    for track, r in results.items():
        spi = r["seconds_per_image"]
        if spi is None:
            print(f"  {track:<14} all images resumed — nothing timed")
            continue
        est = " | ".join(f"n={t}: {spi * t / 3600:5.2f} h" for t in targets)
        print(f"  {track:<14} {spi:6.2f} s/img  ->  {est}")
        for t in targets:
            totals[t] += spi * t / 3600
    print(f"  {'ALL TRACKS':<14} {'':>6}      ->  "
          + " | ".join(f"n={t}: {totals[t]:5.2f} h" for t in targets))
    if str(device) == "cpu":
        print("  ⚠ CPU timing — GPU is far faster; treat as an upper bound.")


def print_runtime_estimate(elapsed, computed, device, targets=(300, 1000, 5000)):
    print("\n" + "=" * 70)
    print("RUNTIME ESTIMATE (from this run's own s/image)")
    print("=" * 70)
    if not computed:
        print("  nothing computed this run (all resumed) — no estimate.")
        return
    spi = elapsed / computed
    print(f"  measured: {spi:.2f} s/image over {computed} image(s) on {device}")
    for t in targets:
        print(f"    n={t:<5} -> {spi * t / 3600:6.2f} h")
    if str(device) == "cpu":
        print("  ⚠ CPU timing — GPU is far faster; treat as an upper bound.")


def finish(args):
    """Last thing every entry point calls: list what is safe to share."""
    return artifact_report(args.output_dir)


# ======================================================================
# 9. The clean-image control experiment — identical for every model.
#
#    Reviewer request: defenses were only ever measured on ATTACKED images, so
#    a reader cannot tell whether they counter the attack or just help/hurt
#    images generally. This measures every defense on CLEAN images. A small
#    negative delta is an expected, publishable result: the clean-accuracy cost.
# ======================================================================
def clean_control(adapter, args, coco_gt, files, image_ids):
    ensembles = load_ensembles()
    names = branch_names(ensembles)
    device = adapter.device
    results = {}

    for track in adapter.tracks:
        is_ocr = track.endswith("ocr")
        sig = rt.pc.config_signature(
            phase=4, experiment="clean_control", track=track,
            model=adapter.model_name, branches=sorted(names),
            ensembles=sorted(ensembles), jpeg_quality=args.jpeg_quality,
            median_kernel=args.median_kernel, tvm_weight=args.tvm_weight,
            tvm_iters=args.tvm_iters, gaussian_sigma=args.gaussian_sigma,
            svd_keep=args.svd_keep_ratio, nms_iou=args.nms_iou,
            n=len(files), ids=ids_digest(image_ids), device=str(device),
            **adapter.sig_fields(args))
        ckpt = make_checkpoint(args, track, sig)

        rt.pc.print_banner(
            f"CLEAN CONTROL — {track} | n={len(files)} | branches={len(names)} "
            f"| ensembles={len(ensembles)}", width=80)

        if is_ocr:
            ocr = adapter.ocr_fn()
            conditions = (["clean_ref", "clean_repeat"]
                          + [f"clean+{n}" for n in names])

            def compute_one(pil_img, img_id, _ocr=ocr):
                texts = {}
                # clean_ref = the UNDEFENDED clean OCR output: the anchor every
                # similarity is measured against.
                texts["clean_ref"] = _ocr(pil_img)
                # A second undefended pass is the determinism check; beam search
                # should reproduce it exactly (=> 1.0).
                texts["clean_repeat"] = _ocr(pil_img)
                defended = build_branches(pil_img, device, args, names)
                for bn in names:
                    texts[f"clean+{bn}"] = _ocr(defended[bn])
                return texts
        else:
            infer = adapter.detection_fn(track)
            conditions = ["clean"] + [f"clean+{n}" for n in names]

            def compute_one(pil_img, img_id, _infer=infer):
                buckets = {}
                record_dets(buckets, "clean", _infer(pil_img), img_id)
                defended = build_branches(pil_img, device, args, names)
                for bn in names:
                    record_dets(buckets, f"clean+{bn}",
                                _infer(defended[bn]), img_id)
                return buckets

        elapsed, computed, failures = run_per_image(
            args, track, files, conditions, compute_one, ckpt)
        ckpt.flush()

        if is_ocr:
            table, per_image = score_ocr(ckpt, names, ensembles,
                                         ref_tag="clean_ref", prefix="clean")
            repeat = sum(text_similarity(t["clean_ref"], t["clean_repeat"])
                         for t in ckpt.buckets.values()) / max(1, len(ckpt.buckets))
            entry = {
                "metric": ("char-level SequenceMatcher similarity of the "
                           "DEFENDED-clean OCR text vs the UNDEFENDED clean OCR "
                           "text. Self-consistency / prediction stability, NOT "
                           "OCR accuracy and NOT a word-recovery ratio."),
                "undefended_clean": 1.0,
                "undefended_clean_note": "1.0 by construction (ref vs itself)",
                "clean_repeat_selfconsistency": repeat,
                "defenses": table,
            }
            write_per_image(args, track,
                            {"note": "per-image similarity vs the undefended "
                                     "clean OCR text", "similarities": per_image})
        else:
            _, conditions_out = score_detection(args, ckpt, names, ensembles,
                                                coco_gt)
            entry = {
                "metric": TRACK_METRIC[track],
                "undefended_clean": conditions_out["clean"]["mAP"],
                "undefended_clean_AP50": conditions_out["clean"]["AP50"],
                "defenses": defense_table(conditions_out, names, ensembles,
                                          "clean", "clean"),
            }
            write_per_image(args, track, {
                "note": "per-image detection COUNTS per clean condition",
                "counts": {str(i): {c: len(b.get(c, [])) for c in conditions}
                           for i, b in ckpt.buckets.items()}})

        entry.update({"num_images": len(ckpt.buckets),
                      "images_computed_this_run": computed,
                      "failures": failures, "seconds": elapsed,
                      "seconds_per_image": elapsed / computed if computed else None})
        results[track] = entry

    # ---------------- report ----------------
    tracks = list(results)
    md = [f"# Clean-image control — {adapter.model_name} "
          f"(defenses on UNATTACKED images)\n"]
    md += header_lines(args, image_ids, extra=[
        "- Metric per track: " + "; ".join(
            f"`{t}` = {TRACK_METRIC[t]}" for t in tracks),
        "- A small negative delta is expected — it is the clean-accuracy cost "
        "of the transform, and it is what tells a reviewer whether a defense "
        "counters the attack or just changes the image.",
    ])
    md.append("## Undefended clean baseline measured by this run\n")
    md.append("| Track | measured | published N=5000 (input) | difference |")
    md.append("|---|---|---|---|")
    for t in tracks:
        u = results[t]["undefended_clean"]
        b = PUBLISHED_BASELINES[t]
        md.append(f"| {t} | {fmt(u)} | {fmt(b)} | {fmt_delta(u - b)} |")
    md.append("")
    md.append("> The published column is an INPUT, not a measurement: "
              "Florence-2 detection comes from the committed N=5000 summaries, "
              "YOLO and OCR from the N=5000 runs done on teammates' machines. "
              "A different n or subset makes the difference a subset effect, "
              "not a discrepancy.\n")

    md.append("## Clean-image performance per defense\n")
    md.append("Δ is versus **this run's own** undefended clean number "
              "(same images, same code path).\n")
    md.append("| Defense | Kind | "
              + " | ".join(f"{t} | Δ" for t in tracks) + " |")
    md.append("|---|---|" + "---|" * (2 * len(tracks)))
    for name in names + list(ensembles):
        label = ENSEMBLE_PAPER_NAMES.get(name) or SOLO_PAPER_NAMES.get(name, name)
        kind = "ENSEMBLE" if name in ensembles else "solo"
        cells = []
        for t in tracks:
            d = results[t]["defenses"][name]
            cells += [fmt(d["value"]), fmt_delta(d["delta_vs_measured"])]
        md.append(f"| {label} (`{name}`) | {kind} | " + " | ".join(cells) + " |")
    md.append("")
    md += timing_table(results, adapter.device)
    md_text = "\n".join(md)

    payload = {
        "phase": 4,
        "experiment": "clean_control",
        "model": adapter.model_name,
        "model_key": args.model_key,
        "purpose": ("clean-image (unattacked) control, so attack-specific "
                    "recovery can be separated from a general effect on images"),
        "smoke_test": bool(args.smoke_test),
        "num_images": args.n,
        "select": args.select,
        "seed": args.seed,
        "image_dir": args.image_dir,
        "ann_file": args.ann_file,
        "image_ids": image_ids,
        "image_ids_sha1_16": ids_digest(image_ids),
        "device": str(adapter.device),
        "branches": names,
        "ensembles": ensembles,
        "ensemble_paper_names": ENSEMBLE_PAPER_NAMES,
        "published_baselines": {
            "note": ("INPUTS to the delta column, not measured here. "
                     "Florence-2 0.3300 from the committed N=5000 summaries; "
                     "YOLO 0.4519 and OCR 1.000 from the N=5000 runs on "
                     "teammates' machines (files not yet in this repo)."),
            **{k: PUBLISHED_BASELINES[k] for k in tracks},
        },
        "config": {k: getattr(args, k) for k in
                   ("jpeg_quality", "median_kernel", "tvm_weight", "tvm_iters",
                    "gaussian_sigma", "svd_keep_ratio", "nms_iou")},
        "model_config": adapter.sig_fields(args),
        "tracks": results,
    }
    stem = f"clean_control_{args.model_key}"
    write_summary(args, payload, stem)
    write_markdown(args, md_text, stem)
    print("\n" + md_text)
    print_timing(results, adapter.device)
    return payload
