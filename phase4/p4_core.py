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
#                                          (--log-name run_<job>.log for
#                                          concurrent jobs in one directory)
#       checkpoint_<track>_<sig8>.pkl      config-keyed, so resume still works
#   Timestamped summaries never overwrite each other; the directory is stable so
#   a crashed run resumes instead of restarting. The per-condition COCO dumps go
#   to a temp dir and are deleted (see p4_logging.py).
# ======================================================================

from __future__ import annotations

import argparse
import hashlib
import io
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

from p4_logging import (DEFAULT_LOG_NAME, artifact_report,  # noqa: E402
                        eval_dump_dir, run_log, valid_log_name)

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
# --all-transforms (clean_control only): the remaining survey transforms of the
# paper's Table 1, so the clean cost exists for every transform whose recovery
# the paper reports. They are phase3_common's own CPU primitives with their
# Phase-3 defaults, exactly as pc.apply_survey_defenses calls them. None is an
# ensemble member. OCR_ONLY mirrors the paper: BM3D and random resize-and-pad
# are reported for OCR only (random resize never maps boxes back, so its
# detection score is a misalignment artefact; BM3D was not run on detection).
EXTRA_PAPER_NAMES = {
    "bit_depth": "Bit-depth",
    "dithering": "Ordered dithering",
    "nlm": "Non-local means",
    "anisotropic": "Anisotropic diffusion",
    "bm3d": "BM3D",
    "random_resize": "Random resize-and-pad",
}
EXTRA_OCR_ONLY = ("bm3d", "random_resize")
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
    p.add_argument("--gpu", type=int, default=-1,
                   help="The N-th device of the VISIBLE set. If "
                        "CUDA_VISIBLE_DEVICES is already set (SLURM, a "
                        "launcher), N indexes into that list and the process "
                        "pins itself to that entry (UUIDs kept as is); if it "
                        "is unset, N is the physical nvidia-smi index. -1 "
                        "(default) = the freest visible GPU.")
    p.add_argument("--log-name", type=_log_name_arg, default=DEFAULT_LOG_NAME,
                   help="File name of the tee'd run log inside --output-dir "
                        "(default run.log). Give concurrent jobs that share "
                        "a directory their own, e.g. run_yolo_s1.log. Must "
                        "match run[A-Za-z0-9_.-]*.log (no path).")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None,
                   help="Default auto (cuda when available). --smoke-test "
                   "falls back to cpu ONLY if you do not set this, so "
                   "`--smoke-test --device cuda` validates the GPU path.")
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
    p.add_argument("--log-every", type=int, default=100,
                   help="Progress heartbeat cadence in images (Phase-3 value).")
    p.add_argument("--verbose-eval", action="store_true",
                   help="Keep pycocotools' full per-condition output in the "
                        "log instead of one summary line per condition.")
    p.add_argument("--skip-clean-consistency", action="store_true",
                   help="OCR: skip the second undefended clean pass (the "
                        "decode-determinism check) and record 1.0 for it. "
                        "Saves ~11%% of the OCR track's runtime.")
    p.add_argument("--deterministic", action="store_true",
                   help="Ask torch for deterministic GPU kernels (cuDNN "
                        "deterministic, benchmark off, "
                        "use_deterministic_algorithms with warn_only), so a "
                        "repeated command can reproduce attacked numbers. An "
                        "EXECUTION flag: not in the checkpoint signature or "
                        "the summary name, recorded in the summary. Ops with "
                        "no deterministic kernel only warn, so verify by "
                        "running one small job twice.")
    p.add_argument("--all-transforms", action="store_true",
                   help="clean_control: also score the other survey transforms "
                        "of the paper (bit-depth, ordered dithering, non-local "
                        "means, anisotropic diffusion; OCR track also BM3D "
                        "and random resize-and-pad). CPU transforms: NLM and "
                        "BM3D add seconds per image. A separate checkpoint "
                        "and summary name (..._alltransforms).")
    p.add_argument("--smoke-test", action="store_true",
                   help="2 images (CPU unless --device is given) into "
                        "<output-dir>/smoke_test, stamped smoke_test=true. "
                        "Proves the plumbing; NOT a result. The only mode "
                        "that tolerates an image dir holding fewer than --n "
                        "images.")
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


def _log_name_arg(text):
    """argparse type for --log-name: a bare run*.log file name, no path."""
    if not valid_log_name(text):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a valid log name: it must be a bare file name "
            f"matching run[A-Za-z0-9_.-]*.log (e.g. run.log, "
            f"run_yolo_s1.log), with no directory part")
    return text


def log_path(args):
    """Where this run tees its console: <output-dir>/<--log-name>."""
    return os.path.join(args.output_dir, args.log_name)


def finalize_args(args, model_key):
    """Resolve the run id, the output dir and the --smoke-test overrides."""
    args.model_key = model_key
    log_name = getattr(args, "log_name", DEFAULT_LOG_NAME)
    if not valid_log_name(log_name):          # argparse already checks it
        raise SystemExit(f"[fatal] invalid --log-name {log_name!r}")
    args.log_name = log_name
    # Timestamp fixed once per process and stamped into every filename it writes.
    args.run_id = time.strftime("%Y%m%d-%H%M%S")
    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.results_root, f"{args.experiment}_{model_key}")
    if args.smoke_test:
        args.n = 2
        args.no_checkpoint = True
        args.output_dir = os.path.join(args.output_dir, "smoke_test")
        # CPU only when the caller did not ask for a device: a smoke test that
        # can never touch CUDA cannot validate the GPU code path.
        if args.device is None:
            args.device = "cpu"
    if args.device is None:
        args.device = "auto"
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


def resolve_repo_file(path):
    """The file to OPEN for a relative asset path such as `yolov8x-worldv2.pt`.

    cwd first (what the library itself would do), then the repo root, so a run
    started from another directory still finds the repo-root copy instead of
    downloading a second one into its cwd. Only the file that is opened
    changes: callers keep the original string for signatures and summaries.
    Absolute paths and paths found nowhere are returned unchanged.
    """
    if not path or os.path.isabs(path) or os.path.isfile(path):
        return path
    at_root = os.path.join(REPO_ROOT, path)
    return at_root if os.path.isfile(at_root) else path


def env_flag(name):
    """True if an environment flag such as HF_HUB_OFFLINE / YOLO_OFFLINE is on."""
    return os.environ.get(name, "").strip().lower() in (
        "1", "true", "yes", "on", "y", "t")


def preflight(args, modules, weights):
    """Shared preflight.

    `modules` = [(name, needed_by)]. `weights` = [(label, present[, where
    [, offline]])]: `where` is the path that was checked (printed), and
    `offline` is a reason string when downloading is impossible (an
    HF_HUB_OFFLINE / YOLO_OFFLINE flag), which turns a missing weight file
    into NOT READY instead of "will download".
    """
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
            path = fn(val)
        except RuntimeError as exc:
            ok = False
            print(f"  {label:<14} MISSING  {exc}")
            continue
        if label != "images":
            print(f"  {label:<14} OK       {path}")
            continue
        # The same count select_images() enforces, so READY means the run can
        # actually start: a short dir is fatal unless --smoke-test.
        k = len(list_images(path))
        if k == 0 or (k < args.n and not args.smoke_test):
            ok = False
            print(f"  {label:<14} SHORT    {path} holds {k} images, --n needs "
                  f"{args.n}. Download {COCO_VAL2017_URL} and unzip it there "
                  f"(or use --smoke-test).")
        else:
            print(f"  {label:<14} OK       {path} ({k} images)")
    print("\n[weights]")
    for item in weights:
        label, present = item[0], item[1]
        where = item[2] if len(item) > 2 else None
        offline = item[3] if len(item) > 3 else None
        if present:
            state = "present/cached"
        elif offline:
            ok = False
            state = f"MISSING  {offline}; no download is possible"
        else:
            state = "will download (needs internet on this node)"
        print(f"  {label:<26} {state}")
        if where:
            print(f"  {'':<26} checked: {where}")
    print("\n[shared core]")
    core = os.path.join(REPO_ROOT, "phase3_common.py")
    if not os.path.isfile(core):
        ok = False
    print(f"  phase3_common.py {'OK' if os.path.isfile(core) else 'MISSING'}"
          f"       {core}")
    print("\n[gpu]")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"  CUDA_VISIBLE_DEVICES={cvd!r} --gpu {args.gpu} --device "
          f"{args.device}: " + (
              "CPU run" if args.device == "cpu" else
              "--gpu indexes into this list" if cvd is not None else
              "--gpu is a physical nvidia-smi index"))
    if args.device != "cpu" and cvd is not None:
        try:      # range check only; the freest pick needs nvidia-smi
            resolve_visible_device(cvd, args.gpu, lambda vis: vis[0])
        except ValueError as exc:
            ok = False
            print(f"  MISSING  {exc}")
    print(f"  run log: {log_path(args)}")
    print("\n[verdict] " + ("READY" if ok else "NOT READY: see MISSING / SHORT above"))
    print("=" * 70)
    return 0 if ok else 1


COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"


def list_images(image_dir):
    """Sorted image filenames in image_dir: what select_images and preflight count."""
    return sorted(f for f in os.listdir(image_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))


def select_images(image_dir, n, seed, mode, allow_short=False):
    # Phase 3 used a bare sorted(os.listdir(...)); the extension filter is a
    # strict improvement that changes nothing on a clean COCO val2017 dir, so
    # `--select head` reproduces exactly the subset the attacked runs scored.
    #
    # Fewer images than --n is an ERROR, not a warning: a "1000-image" run on
    # a partial directory would score a different, smaller subset and still
    # be filed under the requested n. Only --smoke-test (allow_short) keeps
    # the old warning, because it proves plumbing, not results.
    files = list_images(image_dir)
    if not files:
        raise RuntimeError(f"No images found in {image_dir}")
    if len(files) < n and not allow_short:
        raise RuntimeError(
            f"requested n={n} but {image_dir} holds only {len(files)} images: "
            f"the full COCO val2017 (5000 images) is missing. Download "
            f"{COCO_VAL2017_URL}, unzip it so that directory holds all 5000 "
            f"images (or pass --image-dir), and re-run. Use --smoke-test for a "
            f"plumbing check on the images you have.")
    chosen = (files[:n] if mode == "head"
              else sorted(random.Random(seed).sample(files, min(n, len(files)))))
    if len(chosen) < n:
        print(f"[WARN] requested n={n} but only {len(chosen)} images exist; "
              f"treating {len(chosen)} as the requested set (allowed only "
              f"under --smoke-test).")
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
def _query_gpus():
    """[(physical index, uuid, free MiB)] from nvidia-smi; [] if it fails.

    nvidia-smi ignores CUDA_VISIBLE_DEVICES and lists every GPU it can see,
    so the caller restricts the pick to the visible set itself.
    """
    import subprocess
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        gpus = []
        for line in smi.stdout.strip().split("\n"):
            idx, uuid, free = (s.strip() for s in line.split(","))
            gpus.append((int(idx), uuid, int(free)))
        return gpus
    except Exception as exc:  # noqa: BLE001
        print(f"[GPU] nvidia-smi failed ({exc}).")
        return []


def _pick_freest_gpu(visible=None, gpus=None) -> str:
    """The freest GPU, as a CUDA_VISIBLE_DEVICES entry.

    visible=None: among every GPU nvidia-smi lists (returns its physical
    index, the old behaviour). visible=[entries of a preset
    CUDA_VISIBLE_DEVICES]: only among those, matched by physical index or
    by UUID (GPU-...), and the matching ENTRY is returned as written. An
    entry nvidia-smi cannot match (a MIG-... device, or indices that are
    relative to a SLURM cgroup) falls back to the first visible entry.
    `gpus` lets tests pass nvidia-smi's rows instead of running it.
    """
    gpus = _query_gpus() if gpus is None else gpus
    if visible is None:
        if not gpus:
            print("[GPU] no nvidia-smi data; using device 0.")
            return "0"
        return str(max(gpus, key=lambda g: g[2])[0])  # first on ties
    by_index = {str(i): free for i, _, free in gpus}
    by_uuid = {uuid: free for _, uuid, free in gpus}
    scored = []
    for entry in visible:
        free = by_index.get(entry) if entry.isdigit() else by_uuid.get(entry)
        if free is None:
            print(f"[GPU] cannot match visible entry {entry!r} to nvidia-smi; "
                  f"using the first visible entry {visible[0]!r}.")
            return visible[0]
        scored.append((free, entry))
    return max(scored, key=lambda s: s[0])[1]


def resolve_visible_device(preset_cvd, gpu, freest_fn=_pick_freest_gpu):
    """The single CUDA_VISIBLE_DEVICES entry this process pins itself to.

    Pure and torch-free (tested in tests/test_offline_logic.py).
    preset_cvd : CUDA_VISIBLE_DEVICES as found in the environment, None if
                 unset (SLURM and the launcher set it; a shell usually not).
    gpu        : --gpu. N >= 0 is the N-th entry of the preset list, or the
                 physical nvidia-smi index when nothing is preset (the old
                 behaviour). N < 0 picks the freest, among the preset list
                 only when there is one.
    freest_fn  : freest_fn(None | [entries]) -> entry (nvidia-smi based).
    Entries are kept as written, so a UUID (GPU-..., MIG-...) stays a UUID.
    Raises ValueError for an out-of-range N or a preset list with no entry.
    """
    if preset_cvd is None:
        return freest_fn(None) if gpu < 0 else str(gpu)
    visible = [e.strip() for e in preset_cvd.split(",") if e.strip()]
    if not visible:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES={preset_cvd!r} is set but lists no GPU, so "
            f"no GPU is visible to this process. Unset it, or use --device "
            f"cpu.")
    if gpu < 0:
        return visible[0] if len(visible) == 1 else freest_fn(visible)
    if gpu >= len(visible):
        raise ValueError(
            f"--gpu {gpu} is out of range: CUDA_VISIBLE_DEVICES="
            f"{preset_cvd!r} exposes {len(visible)} device(s), so --gpu takes "
            f"0..{len(visible) - 1}, or -1 for the freest of them.")
    return visible[gpu]


def load_runtime(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[GPU] CUDA_VISIBLE_DEVICES='' (CPU run)")
    else:
        preset = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            entry = resolve_visible_device(preset, args.gpu)
        except ValueError as exc:
            raise SystemExit(f"[fatal] {exc}") from exc
        os.environ["CUDA_VISIBLE_DEVICES"] = entry
        if preset is None:
            how = ("freest by nvidia-smi" if args.gpu < 0
                   else "physical nvidia-smi index")
        else:
            how = (f"freest of the preset list {preset!r}" if args.gpu < 0
                   else f"entry {args.gpu} of the preset list {preset!r}")
        print(f"[GPU] CUDA_VISIBLE_DEVICES={entry} ({how}, --gpu {args.gpu})")
    if getattr(args, "deterministic", False):
        # cuBLAS needs this before the first CUDA call to be reproducible.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
    if getattr(args, "deterministic", False):
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
        _torch.use_deterministic_algorithms(True, warn_only=True)
        print("[GPU] --deterministic: cuDNN deterministic, benchmark off, "
              "deterministic algorithms (warn only)")
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


def paper_label(name):
    """Paper label of any branch or ensemble key."""
    return (ENSEMBLE_PAPER_NAMES.get(name) or SOLO_PAPER_NAMES.get(name)
            or EXTRA_PAPER_NAMES.get(name, name))


def extra_branch_names(args, track):
    """The --all-transforms branches of one track ([] without the flag).

    BM3D is an optional dependency: when it cannot be imported the row is
    dropped with a warning, never replaced by another filter (Phase-3 rule).
    """
    if not getattr(args, "all_transforms", False):
        return []
    is_ocr = track.endswith("ocr")
    names = [n for n in EXTRA_PAPER_NAMES
             if is_ocr or n not in EXTRA_OCR_ONLY]
    if "bm3d" in names:
        import importlib.util
        if importlib.util.find_spec("bm3d") is None:
            print("[clean_control] bm3d is not installed: dropping the BM3D "
                  "row (pip install bm3d to include it).")
            names.remove("bm3d")
    return names


def build_extra_branches(pil_img, names, seed=None):
    """{branch: PIL} for the --all-transforms branches.

    The same calls, with the same defaults, as pc.apply_survey_defenses
    (phase3_common.py:416-422); `seed` is the image id, as in Phase 3, so
    random resize-and-pad is reproducible per image.
    """
    pc = rt.pc
    makers = {
        "bit_depth": lambda: pc.bit_depth_cpu(pil_img),
        "dithering": lambda: pc.dithering_cpu(pil_img),
        "nlm": lambda: pc.nlm_cpu(pil_img),
        "anisotropic": lambda: pc.anisotropic_diffusion_cpu(pil_img),
        "bm3d": lambda: pc.bm3d_cpu(pil_img),
        "random_resize": lambda: pc.random_resize_pad_cpu(pil_img, seed=seed),
    }
    return {n: makers[n]() for n in names}


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
                  log_every=None):
    """Run `compute_one(pil, img_id) -> {tag: value}` over `files`.

    Returns (elapsed_seconds, computed_count, failures). `computed` counts only
    images actually processed this run — checkpoint-resumed images cost nothing,
    so dividing by len(files) would report a fake speed and wreck the runtime
    estimate. Asserts, after the loop, that every image carries every condition.
    """
    log_every = log_every or getattr(args, "log_every", 100)
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
    # Flush BEFORE the asserts below: a failed image makes them raise, and the
    # images computed since the last periodic flush would otherwise be lost
    # (up to --checkpoint-every - 1 of them). A re-run then only redoes the
    # failed images.
    ckpt.flush()
    if failures:
        print(f"[{label}] {len(failures)} image(s) failed "
              f"(first: {failures[0]}). Re-run the same command to retry "
              f"them; everything else is checkpointed.")
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
    """COCO-eval every condition. Bulky dumps -> temp dir; output -> one line.

    pycocotools prints ~18 lines per condition (its own progress chatter plus
    the 12-row AP/AR block) and we keep exactly two of those numbers. At 15
    conditions that is ~270 lines of noise in a committed run.log, so the
    boilerplate is captured and dropped, and each condition is logged as one
    line instead. `--verbose-eval` restores the raw output for debugging.
    """
    import contextlib
    with eval_dump_dir(args.output_dir, keep=args.keep_eval_dumps) as dumps:
        if args.verbose_eval:
            stats = rt.pc.evaluate_all_conditions(
                all_results, coco_gt, image_ids=sorted(image_ids),
                output_dir=dumps)
        else:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink):
                stats = rt.pc.evaluate_all_conditions(
                    all_results, coco_gt, image_ids=sorted(image_ids),
                    output_dir=dumps)
    print(f"[eval] {len(stats)} conditions | "
          f"{'mAP':>7} {'AP50':>7}  detections")
    for tag in sorted(stats):
        print(f"[eval]   {float(stats[tag][0]):7.4f} {float(stats[tag][1]):7.4f}"
              f"  {len(all_results.get(tag, [])):>7}  {tag}")
    return stats


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


def header_lines(args, image_ids, extra=(), device=None):
    """Provenance block. `device` is the resolved torch device when known;
    otherwise the --device request is reported."""
    lines = []
    if args.smoke_test:
        dev = device if device is not None else args.device
        lines.append(f"> **SMOKE TEST OUTPUT, NOT A RESULT.** "
                     f"{len(image_ids)} images, device `{dev}`. Do not cite.\n")
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


def print_result_digest(results, names, ensembles, top=3):
    """Compact end-of-run block for the log: baseline + best/worst per track.

    The full per-defense table lives in the committed .md; this is what a person
    watching a tmux pane actually needs.
    """
    print("\n" + "=" * 70)
    print("RESULT DIGEST (full table in the .md next to this log)")
    print("=" * 70)
    for track, r in results.items():
        print(f"  {track}: undefended clean = {r['undefended_clean']:.4f}"
              f"  [{TRACK_METRIC[track]}]")
        rows = sorted(r["defenses"].items(),
                      key=lambda kv: kv[1]["delta_vs_measured"], reverse=True)
        for name, d in rows[:top]:
            label = paper_label(name)
            print(f"      best  {label:<24} {d['value']:.4f} "
                  f"({d['delta_vs_measured']:+.4f})")
        for name, d in rows[-1:]:
            label = paper_label(name)
            print(f"      worst {label:<24} {d['value']:.4f} "
                  f"({d['delta_vs_measured']:+.4f})")


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
    base_names = branch_names(ensembles)
    device = adapter.device
    results = {}

    for track in adapter.tracks:
        is_ocr = track.endswith("ocr")
        extra = extra_branch_names(args, track)
        names = base_names + extra

        def defend(pil_img, img_id, _extra=extra):
            out = build_branches(pil_img, device, args, base_names)
            out.update(build_extra_branches(pil_img, _extra, seed=img_id))
            return out
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
                # A second undefended pass is the decode-determinism check; beam
                # search should reproduce it exactly (=> 1.0). Skippable because
                # it is a whole extra generate per image on the slowest track.
                texts["clean_repeat"] = (
                    texts["clean_ref"] if args.skip_clean_consistency
                    else _ocr(pil_img))
                defended = defend(pil_img, img_id)
                for bn in defended:
                    texts[f"clean+{bn}"] = _ocr(defended[bn])
                return texts
        else:
            infer = adapter.detection_fn(track)
            conditions = ["clean"] + [f"clean+{n}" for n in names]

            def compute_one(pil_img, img_id, _infer=infer):
                buckets = {}
                record_dets(buckets, "clean", _infer(pil_img), img_id)
                defended = defend(pil_img, img_id)
                for bn in defended:
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

        entry.update({"branches": names,
                      "num_images": len(ckpt.buckets),
                      "images_computed_this_run": computed,
                      "failures": failures, "seconds": elapsed,
                      "seconds_per_image": elapsed / computed if computed else None})
        results[track] = entry

    # ---------------- report ----------------
    tracks = list(results)
    md = [f"# Clean-image control — {adapter.model_name} "
          f"(defenses on UNATTACKED images)\n"]
    md += header_lines(args, image_ids, device=device, extra=[
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
    all_names = base_names + [n for n in EXTRA_PAPER_NAMES if any(
        n in results[t]["defenses"] for t in tracks)]
    for name in all_names + list(ensembles):
        label = paper_label(name)
        kind = "ENSEMBLE" if name in ensembles else "solo"
        cells = []
        for t in tracks:
            d = results[t]["defenses"].get(name)
            cells += (["not run", "n/a"] if d is None else
                      [fmt(d["value"]), fmt_delta(d["delta_vs_measured"])])
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
        "branches": base_names,
        "all_transforms": bool(getattr(args, "all_transforms", False)),
        "extra_branches": {t: [n for n in results[t]["branches"]
                               if n not in base_names] for t in tracks},
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
    stem = f"clean_control_{args.model_key}" + (
        "_alltransforms" if getattr(args, "all_transforms", False) else "")
    write_summary(args, payload, stem)
    write_markdown(args, md_text, stem)
    # The full table is already in the .md written above; repeating it here
    # would double it inside the committed run.log for no benefit.
    print_result_digest(results, all_names, ensembles)
    print_timing(results, adapter.device)
    return payload
