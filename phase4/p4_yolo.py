#!/usr/bin/env python3
# ======================================================================
# p4_yolo.py: the YOLOv8x-worldv2 entry point for Phase 4.
#
#   python phase4/p4_yolo.py --experiment clean_control  --n 1000 --gpu 0
#   python phase4/p4_yolo.py --experiment adaptive_bpda  --n 300  --gpu 0
#   python phase4/p4_yolo.py --experiment adaptive_bpda  --target ens_4way
#   python phase4/p4_yolo.py --preflight
#   python phase4/p4_yolo.py --experiment adaptive_bpda --smoke-test
#
# One process, one model, one GPU: designed to be one tmux pane. Everything
# that is not YOLO-specific (CLI, image selection, defense branches, ensemble
# NMS merge, checkpointing, COCO scoring, reports, artifact policy) comes from
# p4_core.py, so it is written once and shared with p4_florence.py.
#
# THIS FILE CONTAINS: the letterbox preprocessing (plus a differentiable copy
# of ultralytics' predict letterbox for --attack-space native), the detection
# parsing, the attack objective (single model and union-aware EOT), the seeded
# PGD with restarts, the BPDA surrogates, and the adaptive BPDA+EOT experiment.
#
# The DEPLOYED defense that every condition is scored with always comes from
# phase3_common via p4_core.build_branches, unchanged. The surrogates below are
# only the attacker's gradient path; their forward values are checked against
# the deployed transforms (assert_*_matches here, tests/test_attack_units.py).
# ======================================================================

from __future__ import annotations

import hashlib
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p4_core as core                                     # noqa: E402
from p4_core import rt                                     # noqa: E402

EXPERIMENTS = ("clean_control", "adaptive_bpda")
OBLIVIOUS_TAG = "pgd_oblivious"
ADAPTIVE_TAG = "pgd_adaptive"
# Per-image attack diagnostics live in the checkpoint bucket under this key.
# assemble_results only reads condition tags, so it never sees it.
META_KEY = "_attack_meta"

# --target choices. The keys are fixed here so argparse works without torch;
# the BRANCH LISTS of the ensembles are read from phase3_common at run time
# (resolve_target), never restated in this file.
ENSEMBLE_TARGETS = ("ens_jpeg_median_gaussian",     # EnsJMG (default)
                    "ens_4way",                     # Ens4W
                    "ens_jpeg_median_tvm",          # EnsJMT
                    "ens_blur_tvm_combo")           # EnsBTC
SOLO_TARGETS = ("median", "jpeg", "gaussian", "tvm", "blur_tvm")
ADAPTIVE_TARGETS = ENSEMBLE_TARGETS + SOLO_TARGETS
DEFAULT_TARGET = "ens_jpeg_median_gaussian"
# The ONE table of BPDA surrogates (section 4): per branch, the backward pass
# ("identity" = straight-through, "gaussian" / "median" = the differentiable
# twin) and the text the summaries print. bpda_branch, surrogate_description,
# is_identity_surrogate and check_surrogates all read it through
# surrogate_spec(). An ensemble whose members are not all in here (e.g.
# EnsJTS: svd) cannot be a --target.
_STRAIGHT_THROUGH = "identity (BPDA straight-through)"
SURROGATES = {
    "jpeg": ("identity", _STRAIGHT_THROUGH),
    "median": ("median", "true a.e. gradient of the unfold median (each output "
                         "routes to the input pixel its window selected)"),
    "gaussian": ("gaussian", "true gradient"),
    "tvm": ("identity", _STRAIGHT_THROUGH),
    "blur_tvm": ("gaussian", "true Gaussian gradient, identity through TVM"),
}
SURROGATE_BRANCHES = tuple(SURROGATES)
# Bumped whenever the attack's numerics change, so it enters the checkpoint
# signature and a stale cache from an older attack is discarded.
PGD_INIT_SCHEME = "sha256(seed|image_id|restart)->cpu-generator-v1"
SURROGATE_VERSION = "bpda-v3"   # v3: bit-exact deployed forward


# ======================================================================
# 1. Letterbox + detection parsing. VERBATIM from Phase 3
#    (archive/Phase3/run_survey_yolo.py:178-226).
# ======================================================================
def letterbox(pil_img, size, fill=(114, 114, 114)):
    w, h = pil_img.size
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = pil_img.resize((nw, nh), rt.Image.BILINEAR)
    canvas = rt.Image.new("RGB", (size, size), fill)
    pl, pt = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pl, pt))
    return canvas, scale, pl, pt, nw, nh


def unletterbox(lb_np, orig_size, pl, pt, nw, nh):
    cropped = lb_np[pt:pt + nh, pl:pl + nw]
    return rt.Image.fromarray(cropped).resize(orig_size, rt.Image.BICUBIC)


def letterbox_differentiable(x, size, stride=32, auto=True, fill=114 / 255.0):
    """Differentiable copy of ultralytics' predict-time LetterBox.

    ultralytics' `model.predict` builds `LetterBox(imgsz, auto=rect, stride)`
    with rect=True (verified in ultralytics 8.4.127: engine/model.py predict,
    engine/predictor.py pre_transform, data/augment.py LetterBox.get_params).
    This repeats that geometry exactly (round() for the resized size,
    minimum-rectangle padding to the stride, centred split with the -0.1/+0.1
    rounding, grey 114) on a float [1,3,H,W] tensor in [0,1], with
    F.interpolate(bilinear) standing in for cv2.INTER_LINEAR. With no resize
    (longest side == size, most COCO images) it is exact; otherwise the two
    interpolators differ by rounding, under 1/255 per pixel on the
    test images (tests/test_attack_units.py compares against ultralytics'
    own LetterBox).
    """
    F = rt.torch.nn.functional
    h, w = int(x.shape[-2]), int(x.shape[-1])
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = size - nw, size - nh
    if auto:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    if (nh, nw) != (h, w):
        x = F.interpolate(x, size=(nh, nw), mode="bilinear",
                          align_corners=False)
    return F.pad(x, [left, right, top, bottom], value=fill)


class YoloAdapter:
    """Loads YOLOv8x-worldv2 once and exposes inference to p4_core."""

    tracks = ("yolo",)

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model_name = args.yolo_model
        self.model = None
        self._yolo_to_coco_id = {}

    def sig_fields(self, args):
        return {"yolo_model": args.yolo_model, "imgsz": args.yolo_imgsz,
                "conf": args.yolo_conf, "yolo_nms_iou": args.yolo_iou}

    def load(self, coco_gt):
        from ultralytics import YOLO
        cats = sorted(coco_gt.loadCats(coco_gt.getCatIds()),
                      key=lambda c: c["id"])
        names = [c["name"] for c in cats]
        self._yolo_to_coco_id = {i: c["id"] for i, c in enumerate(cats)}
        self.model = YOLO(self.args.yolo_model)
        self.model.set_classes(names)
        self.model.to(self.device)
        self.model.model.eval()
        print(f"[yolo] loaded {self.args.yolo_model}: {len(names)} COCO "
              f"classes | imgsz={self.args.yolo_imgsz} | device={self.device}")

    def _parse(self, res):
        dets = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            cls_idx = int(clss[i])
            if cls_idx not in self._yolo_to_coco_id:
                continue
            dets.append({"bbox": [float(x1), float(y1), float(w), float(h)],
                         "category_id": self._yolo_to_coco_id[cls_idx],
                         "score": float(confs[i])})
        return dets

    def detection_fn(self, track="yolo"):
        a = self.args

        def run_inference(pil_img):
            res = self.model.predict(pil_img, conf=a.yolo_conf, iou=a.yolo_iou,
                                     imgsz=a.yolo_imgsz, verbose=False,
                                     device=self.device)
            return self._parse(res[0])

        return run_inference

    def ocr_fn(self):
        raise RuntimeError("YOLO has no OCR track.")


# ======================================================================
# 2. The attack objective.
#
# YOLO EMITS NO TOKENS: there is no cross-entropy here, and the paper must not
# describe it as one. `pred = model.model(x)[0]` is [1, 4+80, A] (A = 8400 for
# a 640x640 input); the class block `pred[:, 4:, :]` holds POST-SIGMOID class
# confidences (ultralytics' WorldDetect head concatenates `cls.sigmoid()` onto
# the decoded boxes). Per anchor we take the max over the 80 classes; the
# SUPPRESSION OBJECTIVE J is the sum of those over all anchors, and PGD
# descends it (lower J = more suppression). Untargeted; no ground truth, no
# self-labels, and the box block pred[:, :4, :] is excluded. YOLOv8 is
# anchor-free, so there is no objectness term to include.
# Phase-3 objective: archive/Phase3/run_survey_yolo.py:243-247 (its loss is
# exactly -J; see yolo_confidence_loss).
# ======================================================================
def anchor_scores_from_pred(pred):
    """[B, 4+C, A] head output -> [B, A] max-class confidence per anchor."""
    return pred[:, 4:, :].max(dim=1)[0]


def yolo_anchor_scores(model, x):
    # The Detect head caches anchor tensors keyed on input shape; they are not
    # part of a new autograd graph, so the cache must be reset before EVERY
    # forward (archive/Phase3/run_survey_yolo.py:243/266/303).
    model.model.model[-1].shape = None
    preds = model.model(x)
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    return anchor_scores_from_pred(pred)


def yolo_confidence_loss(model, x):
    """The Phase-3 loss, kept for reference: exactly -J for one model input."""
    return -yolo_anchor_scores(model, x).sum()


def union_objective(scores, mode):
    """EOT objective over k branches. `scores` is [k, B, A]; lower = stronger.

    mean : mean over branches of each branch's J (the Phase-4 pilot's loss).
    max  : union-aware. The NMS-union merge keeps a box if ANY branch keeps it,
           so per anchor only the most confident branch matters: sum over
           anchors of the max over branches of the max-class confidence.
           Anchors align across branches because every branch reaches the
           model at the same input size.
    """
    if mode == "mean":
        return scores.sum(dim=(1, 2)).mean()
    if mode == "max":
        return scores.max(dim=0).values.sum()
    raise ValueError(f"unknown EOT loss '{mode}'")


def union_weights(scores):
    """One-hot [k, B, A] marking the winning branch of every anchor.

    d(max-objective)/d(scores) is exactly this mask (ties have measure zero),
    which is what lets the max objective's gradient be taken one branch at a
    time instead of keeping k model graphs alive at once.
    """
    w = rt.torch.zeros_like(scores)
    w.scatter_(0, scores.argmax(dim=0, keepdim=True), 1.0)
    return w


# ======================================================================
# 3. Seeded PGD init, quantization, and the attack frame.
# ======================================================================
def pgd_init_seed(seed, image_id, restart=0):
    """Deterministic 63-bit seed for one (run seed, image, restart) triple.

    Stdlib only, so the derivation itself is testable without torch. Restart r
    always gets the same draw, whatever --restarts is set to.
    """
    blob = f"pgd-init|{int(seed)}|{int(image_id)}|{int(restart)}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") >> 1


def pgd_init_noise(x0, eps, seed, image_id, restart=0):
    """uniform(-eps, eps) noise shaped like x0, from a per-image generator.

    Drawn on a CPU generator and then moved, so the draw is the same on CPU
    and CUDA. Both attacks call this with the same arguments, which is what
    makes them start from the SAME point (Phase 3 used the unseeded global
    RNG, so its oblivious and adaptive draws differed).
    """
    torch = rt.torch
    g = torch.Generator(device="cpu").manual_seed(
        pgd_init_seed(seed, image_id, restart))
    u = torch.rand(tuple(x0.shape), generator=g, dtype=torch.float32)
    return ((u * 2.0 - 1.0) * eps).to(device=x0.device, dtype=x0.dtype)


def quantize_array(arr255, mode):
    """float array already scaled to [0,255] -> uint8.

    truncate : `.astype(uint8)`, the Phase-3 conversion (floors every value).
    round    : nearest integer.
    """
    np = rt.np
    if mode == "round":
        return np.rint(arr255).astype(np.uint8)
    if mode == "truncate":
        return arr255.astype(np.uint8)
    raise ValueError(f"unknown quantize mode '{mode}'")


def uint8_budget(eps):
    """Largest whole uint8 step inside an L_inf budget eps (0.03 -> 7)."""
    return int(math.floor(eps * 255 + 1e-6))


def quantize_adv(adv, mode, x0=None, eps=None):
    """[1,3,H,W] in [0,1] -> (uint8 HWC array, the same values as a tensor).

    With x0 (exact uint8/255, true of both attack frames) and eps, the uint8
    values are also clipped to x0*255 +/- uint8_budget(eps). eps*255 is
    rarely whole (0.03 -> 7.65): truncation floors -7.65 to -8 and rounding
    gives +/-8, i.e. 8/255 = 0.0314 > eps.
    """
    np, torch = rt.np, rt.torch
    u8 = quantize_array(
        adv.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255, mode)
    if x0 is not None:
        k = uint8_budget(eps)
        ref = np.rint(x0.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
                      * 255).astype(np.int16)
        u8 = np.clip(u8.astype(np.int16), ref - k, ref + k).astype(np.uint8)
    adv_q = torch.from_numpy(
        u8.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
    return u8, adv_q.to(adv.device)


def _model_stride(model, default=32):
    stride = getattr(getattr(model, "model", None), "stride", None)
    if stride is None:
        return default
    return int(rt.torch.as_tensor(stride).max())


def attack_frame(pil_img, device, args, model=None):
    """Where the PGD variable lives and how it reaches the model.

    letterbox (default, Phase-3 compatible): the variable is the 640x640
        square letterbox, fed to the model as is, and un-letterboxed (bicubic)
        to native resolution after the attack.
    native: the variable is the native-resolution image (exactly what the
        deployed defense receives); the model input is letterbox_differentiable
        of it, and the result needs no resampling.
    """
    torch, np = rt.torch, rt.np
    if args.attack_space == "native":
        size, stride = args.yolo_imgsz, _model_stride(model)
        return SimpleNamespace(
            space="native", x0=rt.pc.pil_to_tensor(pil_img, device),
            to_model=lambda t: letterbox_differentiable(t, size, stride),
            to_pil=lambda u8: rt.Image.fromarray(u8))
    orig_size = pil_img.size
    lb, _, pl, pt, nw, nh = letterbox(pil_img, args.yolo_imgsz)
    img_np = np.array(lb).astype(np.float32) / 255.0
    x0 = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return SimpleNamespace(
        space="letterbox", x0=x0, to_model=lambda t: t,
        to_pil=lambda u8: unletterbox(u8, orig_size, pl, pt, nw, nh))


# ======================================================================
# 4. BPDA surrogates (Athalye, Carlini & Wagner, ICML 2018).
#
# Every surrogate FORWARDS the exact deployed value (the phase3_common
# transform, including the uint8 round-trips the deployed PIL pipeline does)
# and differs only in the BACKWARD pass:
#   jpeg, tvm        identity Jacobian (straight-through)
#   median           true a.e. gradient of the unfold median (default), or
#                    identity with --median-surrogate identity
#   gaussian         true gradient
#   blur_tvm         true Gaussian gradient, identity through TVM
# The uint8 round-trips are straight-through in every case. The per-branch
# choice is read from SURROGATES (top of file) via surrogate_spec().
# ======================================================================
def gaussian_differentiable(x, sigma):
    """Grad-enabled twin of pc.gaussian_gpu (phase3_common.py:144-165).

    pc.gaussian_gpu is @torch.no_grad()-decorated, so it cannot be
    differentiated. This repeats its exact op sequence (normalised 1-D kernel,
    reflect pad, two grouped convs, clamp) with autograd on.
    assert_gaussian_matches() proves at startup that the two agree, so this is
    a gradient path, not a second implementation.
    """
    torch, np, F = rt.torch, rt.np, rt.torch.nn.functional
    radius = int(np.ceil(3 * sigma))
    size = 2 * radius + 1
    coords = torch.arange(size, device=x.device, dtype=x.dtype) - radius
    k1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    k1d = k1d / k1d.sum()
    C = x.shape[1]
    kh = k1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    y = F.pad(x, [radius, radius, 0, 0], mode="reflect")
    y = F.conv2d(y, kh, groups=C)
    kv = k1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    y = F.pad(y, [0, 0, radius, radius], mode="reflect")
    y = F.conv2d(y, kv, groups=C)
    return y.clamp(0, 1)


def median_differentiable(x, kernel_size):
    """Grad-enabled twin of pc.median_gpu (phase3_common.py:121-128).

    Same reflect padding, same unfold, same torch.median, no @no_grad. The
    median is differentiable almost everywhere: each output pixel's gradient
    goes to the one input pixel its window selected (zero for the rest).
    """
    F = rt.torch.nn.functional
    N, C, H, W = x.shape
    pad = kernel_size // 2
    y = F.pad(x, [pad] * 4, mode="reflect")
    patches = y.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    return patches.contiguous().view(N, C, H, W, -1).median(dim=-1).values


def _probe(device):
    g = rt.torch.Generator(device="cpu").manual_seed(0)
    return rt.torch.rand(1, 3, 64, 64, generator=g).to(device)


def assert_gaussian_matches(device, sigma, tol=1e-6):
    """Refuse to run if the BPDA Gaussian differs from the deployed one."""
    probe = _probe(device)
    diff = (rt.pc.gaussian_gpu(probe, sigma=sigma)
            - gaussian_differentiable(probe, sigma)).abs().max().item()
    assert diff < tol, (
        f"gaussian_differentiable deviates from pc.gaussian_gpu by {diff:.2e} "
        f"(> {tol:.0e}); the BPDA gradient would not match the deployed "
        f"defense. Refusing to run.")
    print(f"[guard] gaussian_differentiable == pc.gaussian_gpu "
          f"(max |diff| = {diff:.2e})")


def assert_median_matches(device, kernel_size, tol=1e-7):
    """Refuse to run if the differentiable median differs from the deployed one."""
    probe = _probe(device)
    diff = (rt.pc.median_gpu(probe, kernel_size)
            - median_differentiable(probe, kernel_size)).abs().max().item()
    assert diff < tol, (
        f"median_differentiable deviates from pc.median_gpu by {diff:.2e}; "
        f"refusing to run.")
    print(f"[guard] median_differentiable == pc.median_gpu "
          f"(max |diff| = {diff:.2e})")


def surrogate_spec(branch, args):
    """(backward kind, description) of a branch, from SURROGATES.

    --median-surrogate identity swaps the median's true gradient for the
    straight-through one; nothing else depends on the flags.
    """
    if branch not in SURROGATES:
        raise ValueError(f"no BPDA surrogate for branch '{branch}'")
    if branch == "median" and args.median_surrogate == "identity":
        return "identity", _STRAIGHT_THROUGH
    return SURROGATES[branch]


def check_surrogates(device, args, branches):
    """Run the forward-equality guards the chosen branches depend on."""
    kinds = {surrogate_spec(b, args)[0] for b in branches}
    if "gaussian" in kinds:
        assert_gaussian_matches(device, args.gaussian_sigma)
    if "median" in kinds:
        assert_median_matches(device, args.median_kernel)


def _quantize_uint8(t):
    """The PIL uint8 round-trip the deployed pipeline performs (truncating)."""
    return rt.pc.pil_to_tensor(rt.pc.tensor_to_pil(t), t.device)


def deployed_value(x, branch, args):
    """The deployed transform of a [1,3,H,W] tensor, no gradient.

    Mirrors build_branches / pc.apply_all_defenses_gpu op for op, including
    where the deployed pipeline quantizes (PIL in, PIL out) and where it does
    not (blur_tvm feeds the UNquantized Gaussian into TVM).
    """
    pc, q = rt.pc, _quantize_uint8
    with rt.torch.no_grad():
        if branch == "jpeg":
            return pc.pil_to_tensor(
                pc.jpeg_cpu(pc.tensor_to_pil(x), args.jpeg_quality), x.device)
        if branch == "median":
            return q(pc.median_gpu(q(x), args.median_kernel))
        if branch == "gaussian":
            return q(pc.gaussian_gpu(q(x), sigma=args.gaussian_sigma))
        if branch == "tvm":
            return q(pc.tvm_gpu(q(x), weight=args.tvm_weight,
                                n_iter=args.tvm_iters))
        if branch == "blur_tvm":
            return q(pc.tvm_gpu(pc.gaussian_gpu(q(x), sigma=args.gaussian_sigma),
                                weight=args.tvm_weight, n_iter=args.tvm_iters))
    raise ValueError(f"no BPDA surrogate for branch '{branch}'")


def bpda_branch(adv, branch, args, value=None):
    """Forward = the REAL (deployed) transform. Backward = the surrogate.

    `value + (s - s.detach())` forwards `value` bit-exactly (s - s is exactly
    0 in IEEE arithmetic) while d(out)/d(adv) = ds/dadv; with s = adv that is
    the BPDA identity. The textbook `s + (value - s).detach()` has the same
    gradient but forwards `value` only up to float32 rounding.
    """
    kind = surrogate_spec(branch, args)[0]
    if value is None:
        value = deployed_value(adv.detach(), branch, args)
    if kind == "gaussian":
        s = gaussian_differentiable(adv, args.gaussian_sigma)
    elif kind == "median":
        s = median_differentiable(adv, args.median_kernel)
    else:
        s = adv
    return value.detach() + (s - s.detach())


def surrogate_description(branch, args):
    return surrogate_spec(branch, args)[1]


def is_identity_surrogate(branch, args):
    return surrogate_spec(branch, args)[0] == "identity"


# ======================================================================
# 5. Objectives and the PGD loop shared by both attacks.
#    An objective is fn(adv, need_grad) -> (J as float, dJ/dadv or None).
# ======================================================================
def oblivious_objective(model, frame):
    """J of the bare model: the attacker does not know a defense exists."""
    torch = rt.torch

    def objective(adv, need_grad):
        if not need_grad:
            with torch.no_grad():
                return float(yolo_anchor_scores(
                    model, frame.to_model(adv)).sum()), None
        x = adv.detach().requires_grad_(True)
        J = yolo_anchor_scores(model, frame.to_model(x)).sum()
        (g,) = torch.autograd.grad(J, x)
        return float(J.detach()), g

    return objective


def eot_objective(model, frame, branches, args):
    """J through the defense branches, aggregated by --eot-loss.

    The deployed transform of each branch is computed ONCE per call and reused
    by the gradient pass. For `max` with k > 1 branches a no-grad pass first
    picks the winning branch per anchor; each branch's graph is then built and
    freed in turn with that branch's anchors as the mask. The gradient equals
    the single-backward autograd of union_objective(max) over the k stacked
    graphs (tests/test_attack_units.py checks this), but only one YOLOv8x
    graph is alive at a time instead of k. That is a memory choice for the
    6 GB card; it costs one extra no-grad forward per branch per step.
    """
    torch = rt.torch
    use_max = args.eot_loss == "max" and len(branches) > 1

    def scores_of(values):
        with torch.no_grad():
            return torch.stack([yolo_anchor_scores(model, frame.to_model(v))
                                for v in values])

    def objective(adv, need_grad):
        adv = adv.detach()
        values = [deployed_value(adv, b, args) for b in branches]
        weights = None
        if not need_grad or use_max:
            S = scores_of(values)
            value = float(union_objective(S, args.eot_loss))
            if not need_grad:
                return value, None
            weights = union_weights(S)
        grads, vals = [], []
        for i, b in enumerate(branches):
            if weights is not None and not bool(weights[i].any()):
                continue                       # this branch wins no anchor
            x = adv.clone().requires_grad_(True)
            s = yolo_anchor_scores(
                model, frame.to_model(bpda_branch(x, b, args, values[i])))
            J = s.sum() if weights is None else (s * weights[i]).sum()
            (g,) = torch.autograd.grad(J, x)
            grads.append(g.detach())
            vals.append(float(J.detach()))
        if weights is None:                    # mean: EOT over the branches
            return sum(vals) / len(vals), torch.stack(grads).mean(dim=0)
        return value, torch.stack(grads).sum(dim=0)

    return objective


def run_pgd(frame, objective, args, image_id):
    """L_inf PGD with --restarts seeded restarts; keeps the lowest final J.

    Update rule, projection and clamps are Phase 3's
    (archive/Phase3/run_survey_yolo.py:255-280): `adv + alpha*sign(dLoss)`
    with Loss = -J is written here as `adv - alpha*sign(dJ)`, the same step.
    Each restart's result is quantized (--quantize) and its J re-measured on
    the quantized tensor, which is what the selection compares. In native
    space the uint8 result is clipped back inside eps (quantize_adv), since
    it is exactly what the defense receives; letterbox keeps Phase 3's
    conversion (its output is resampled to native size, so no per-pixel
    bound survives there anyway).
    """
    torch = rt.torch
    x0 = frame.x0
    best, restart_objs = None, []
    for r in range(args.restarts):
        adv = (x0 + pgd_init_noise(x0, args.eps, args.seed, image_id, r)
               ).clamp(0.0, 1.0)
        for _ in range(args.pgd_iters):
            _, grad = objective(adv, need_grad=True)
            with torch.no_grad():
                adv = adv - args.alpha * grad.sign()
                delta = torch.clamp(adv - x0, min=-args.eps, max=args.eps)
                adv = torch.clamp(x0 + delta, 0.0, 1.0).detach()
        u8, adv_q = quantize_adv(
            adv, args.quantize, x0 if frame.space == "native" else None,
            args.eps)
        obj, _ = objective(adv_q, need_grad=False)
        restart_objs.append(obj)
        if best is None or obj < best[0]:
            best = (obj, r, u8, adv_q)
    obj, r, u8, adv_q = best
    return frame.to_pil(u8), {"objective": obj, "restart": r,
                              "restart_objectives": restart_objs,
                              "adv_q": adv_q}


def pgd_oblivious(pil_img, model, device, args, image_id):
    """Phase-3 PGD on the bare model (the attacker ignores the defense).

    Returns (adversarial PIL at native size, info). With the defaults
    (letterbox space, truncate, 1 restart) it differs from Phase 3 only in
    the seeded init.
    """
    frame = attack_frame(pil_img, device, args, model)
    return run_pgd(frame, oblivious_objective(model, frame), args, image_id)


def pgd_adaptive_bpda_eot(pil_img, model, device, args, branches, image_id):
    """PGD whose objective runs THROUGH the defense branches (BPDA + EOT).

    Same frame, same seeded init and same step rule as pgd_oblivious, so the
    two differ only in the gradient they follow.
    """
    frame = attack_frame(pil_img, device, args, model)
    return run_pgd(frame, eot_objective(model, frame, branches, args), args,
                   image_id)


def target_objectives(pil_img, model, device, args, branches, adv_q_by_name):
    """J through the target defense for the clean image and each attack output.

    `adv_q_by_name` maps a name to a quantized tensor in the attack frame (or
    None for the clean image). The adaptive attack should reach a LOWER J than
    the oblivious one on the same image; if it does not, it is not working.
    """
    frame = attack_frame(pil_img, device, args, model)
    objective = eot_objective(model, frame, branches, args)
    return {name: objective(frame.x0 if t is None else t, need_grad=False)[0]
            for name, t in adv_q_by_name.items()}


# ======================================================================
# 6. Targets and the disclosure list.
# ======================================================================
def resolve_target(target, table=None):
    """(ensembles, branches, paper_name) for --target.

    Ensemble members are read from phase3_common.TIER1_ENSEMBLES (which is
    ENSEMBLES plus ens_4way / ens_jpeg_tvm_svd), never restated here. `table`
    lets the torch-free tests pass the AST-parsed constant instead.
    """
    if table is None:
        table = rt.pc.TIER1_ENSEMBLES
    if target in ENSEMBLE_TARGETS:
        assert target in table, (
            f"--target {target} is not in phase3_common.TIER1_ENSEMBLES")
        branches = list(table[target])
        ensembles = {target: branches}
    elif target in SOLO_TARGETS:
        branches, ensembles = [target], {}
    else:
        raise ValueError(f"unknown --target '{target}'")
    missing = [b for b in branches if b not in SURROGATE_BRANCHES]
    assert not missing, (
        f"--target {target} has members {missing} with no BPDA surrogate")
    return ensembles, branches, paper_name(target)


def paper_name(key):
    """Paper label of a branch or ensemble key (EnsJMG, Ens4W, Median, ...)."""
    return (core.ENSEMBLE_PAPER_NAMES.get(key)
            or core.SOLO_PAPER_NAMES.get(key, key))


ATTACK_LABELS = {"clean": "clean", OBLIVIOUS_TAG: "oblivious attack",
                 ADAPTIVE_TAG: "adaptive attack"}


def condition_label(tag):
    """'pgd_adaptive+ens_4way' -> ('adaptive attack', 'Ens4W')."""
    prefix, _, defense = tag.partition("+")
    return (ATTACK_LABELS.get(prefix, prefix),
            paper_name(defense) if defense else "undefended")


ATTACK_FLAGS = ("target", "eot_loss", "restarts", "median_surrogate",
                "quantize", "attack_space")
# parse_args defaults of the ablation flags (tests/test_offline_logic.py
# checks they match); a non-default value is appended to the summary stem.
STEM_FLAG_DEFAULTS = {"eot_loss": "max", "restarts": 1,
                      "median_surrogate": "true", "quantize": "truncate",
                      "attack_space": "letterbox"}


def summary_stem(args):
    """One summary stem per (target, seed, non-default ablation flags).

    With --target selectable and 3 seeds per target, a single
    `summary_adaptive_bpda_yolo_latest.json` would silently mirror whichever
    configuration ran last. Keying the stem on all of them keeps every
    configuration's `_latest` stable to cite inside the one resumable run
    directory (checkpoints are already separated by their signature): an
    ablation such as `--median-surrogate identity` gets
    `..._seed0_median-surrogate-identity`, never the headline's `_latest`.
    Eps and iteration diagnostics still need their own --output-dir.
    """
    stem = f"adaptive_bpda_{args.model_key}_{args.target}_seed{args.seed}"
    for k, default in STEM_FLAG_DEFAULTS.items():
        if getattr(args, k) != default:
            stem += f"_{k.replace('_', '-')}-{getattr(args, k)}"
    return stem


def eot_description(args, branches):
    if len(branches) == 1:
        return ("single branch: EOT is trivial and --eot-loss has no effect")
    if args.eot_loss == "mean":
        return (f"mean of the {len(branches)} branch objectives (per-step "
                f"gradient = mean of the branch gradients)")
    return (f"union-aware max: per anchor, max over the {len(branches)} "
            f"branches of the max-class confidence, summed over anchors")


def implementation_uncertainty(args, paper, branches):
    """What a reader must know about this attack, for the chosen options."""
    sz = args.yolo_imgsz
    items = []
    if args.attack_space == "letterbox":
        items += [
            f"Scale mismatch (still present in letterbox space): the attack "
            f"optimizes a {sz}x{sz} square letterbox while the deployed defense "
            f"runs on the full-resolution un-letterboxed image, so every BPDA "
            f"surrogate applies its transform at a different scale than "
            f"deployment. The adaptive damage reported here is therefore a "
            f"LOWER bound on a fully matched attack; --attack-space native "
            f"removes this mismatch.",
            "Geometry mismatch: ultralytics' model.predict, which scores every "
            "condition, letterboxes with rect=True (minimum-rectangle padding "
            "to the model stride, so a 640x480 image is fed as 640x480; checked "
            "in ultralytics 8.4.127), while this attack feeds the model a "
            "square canvas with grey bars. Padding and anchor grid differ "
            "between the attacked and the scored input, as in every Phase-3 "
            "YOLO attack.",
            f"The adversarial tensor is bicubic-resampled back to native "
            f"resolution outside the gradient path (as in every Phase-3 "
            f"attack), which partially destroys the perturbation whenever the "
            f"image's longest side is not {sz}.",
            f"No per-pixel L_inf bound holds at native resolution after that "
            f"resample, and the uint8 conversion of the letterbox can already "
            f"move a pixel by {math.ceil(args.eps * 255 - 1e-6)}/255 (eps*255 "
            f"= {args.eps * 255:g}), as in Phase 3; --attack-space native "
            f"clips to eps.",
        ]
    else:
        items.append(
            f"Native space: the perturbation lives at native resolution, each "
            f"surrogate defense runs there as deployed, and the model input is "
            f"a differentiable copy of ultralytics' predict letterbox "
            f"(minimum-rectangle padding to the stride, grey 114, centred). It "
            f"resizes with torch bilinear where ultralytics uses cv2 "
            f"INTER_LINEAR on uint8, so the two agree exactly when no resize "
            f"is needed (longest side == {sz}) and differ by interpolation "
            f"rounding otherwise (under 1/255 per pixel against ultralytics' "
            f"LetterBox in tests/test_attack_units.py). The letterbox-vs-native "
            f"scale mismatch of the default mode is removed and nothing "
            f"resamples the perturbation after the attack. The uint8 output "
            f"is clipped to the clean image +/- {uint8_budget(args.eps)}/255 "
            f"(eps*255 = {args.eps * 255:g} is not whole, and truncation or "
            f"rounding alone would move a pixel by "
            f"{math.ceil(args.eps * 255 - 1e-6)}/255), so no pixel the "
            f"defense receives moves more than eps.")
    ident = [b for b in branches if is_identity_surrogate(b, args)]
    true = [b for b in branches if not is_identity_surrogate(b, args)]
    if ident:
        items.append(
            f"BPDA identity Jacobian (straight-through) for "
            f"{', '.join(ident)}: the standard approximation, not the "
            f"transform's true Jacobian; a learned differentiable proxy could "
            f"give a stronger attacker.")
    if true:
        items.append(
            "True gradients: " + "; ".join(
                f"{b}: {surrogate_description(b, args)}" for b in true)
            + ". The median gradient is sparse (one input per output pixel), "
              "so it carries less signal per step than a smooth filter's.")
    items.append(
        "Every surrogate forwards the deployed phase3_common transform's exact "
        "output on the attack-frame tensor ("
        + ("the letterbox, so only up to the scale mismatch above"
           if args.attack_space == "letterbox" else "the native image")
        + "), including the deployed pipeline's truncating uint8 round-trips "
        "(phase3_common.tensor_to_pil); those round-trips are straight-through "
        "in the backward pass. "
        + ("The attack output is converted to uint8 by truncation, the "
           "Phase-3 conversion: every perturbed pixel is floored, on average "
           "0.5/255 below rounding." if args.quantize == "truncate" else
           "The attack output is rounded to uint8 (Phase 3 truncated)."))
    if len(branches) > 1 and args.eot_loss == "max":
        items.append(
            "EOT loss 'max' mirrors the NMS-union merge (a box survives if ANY "
            "branch keeps it) by suppressing the most confident branch per "
            "anchor. It is still a proxy: the real merge matches "
            "post-threshold boxes across branches by IoU, not anchors by "
            "index. Its gradient is exact but taken branch by branch (winner "
            "per anchor picked in a no-grad pass).")
    elif len(branches) > 1:
        items.append(
            f"EOT loss 'mean' averages the {len(branches)} branch objectives; "
            f"it ignores that the NMS-union merge keeps a box if ANY branch "
            f"keeps it, so a box one branch still detects is not targeted "
            f"specifically.")
    items.append(
        f"EOT aggregates over deterministic defense branches, not over defense "
        f"randomness ({paper} has none); a randomized defense would need EOT "
        f"samples per branch as well.")
    items.append(
        "PGD init: one seeded uniform(-eps, eps) draw per (seed, image id, "
        "restart), identical for the oblivious and the adaptive attack, so "
        "the two differ only in the gradient they follow. Phase 3 drew its "
        "init from the unseeded global RNG, so Phase-3 and Phase-4 attacked "
        "numbers also differ by init noise.")
    if args.restarts > 1:
        items.append(
            f"{args.restarts} restarts per image: the kept one has the lowest "
            f"surrogate objective (oblivious: bare model; adaptive: through "
            f"{paper}) on the quantized {args.attack_space} tensor, not the "
            f"lowest defended mAP, which would need ground truth.")
    return items


def adaptive_signature(args, adapter, target, branches, files, image_ids,
                       device):
    """Checkpoint signature: every flag that changes an attacked image."""
    return rt.pc.config_signature(
        phase=4, experiment="adaptive_bpda", model=adapter.model_name,
        target=target, branches=sorted(branches), eps=args.eps,
        iters=args.pgd_iters, alpha=args.alpha, seed=args.seed,
        init=PGD_INIT_SCHEME, surrogates=SURROGATE_VERSION,
        restarts=args.restarts, eot_loss=args.eot_loss,
        median_surrogate=args.median_surrogate, quantize=args.quantize,
        attack_space=args.attack_space, jpeg_quality=args.jpeg_quality,
        median_kernel=args.median_kernel, gaussian_sigma=args.gaussian_sigma,
        tvm_weight=args.tvm_weight, tvm_iters=args.tvm_iters,
        nms_iou=args.nms_iou, n=len(files), ids=core.ids_digest(image_ids),
        device=str(device), **adapter.sig_fields(args))


# ======================================================================
# 7. The adaptive white-box experiment (YOLO only: no beam search, so it is
#    cheap enough to run a real PGD through every branch per image).
# ======================================================================
def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def adaptive_bpda(adapter, args, coco_gt, files, image_ids):
    target = args.target
    ensembles, branches, paper = resolve_target(target)
    is_ens = bool(ensembles)
    device, model = adapter.device, adapter.model
    infer = adapter.detection_fn("yolo")
    check_surrogates(device, args, branches)

    conditions = (["clean"] + [f"clean+{b}" for b in branches]
                  + [OBLIVIOUS_TAG] + [f"{OBLIVIOUS_TAG}+{b}" for b in branches]
                  + [ADAPTIVE_TAG] + [f"{ADAPTIVE_TAG}+{b}" for b in branches])
    sig = adaptive_signature(args, adapter, target, branches, files, image_ids,
                             device)
    ckpt = core.make_checkpoint(args, "adaptive", sig)

    rt.pc.print_banner(
        f"ADAPTIVE WHITE-BOX: BPDA+EOT vs {paper} | n={len(files)} | PGD "
        f"eps={args.eps} x{args.pgd_iters} | R={args.restarts} | "
        f"eot={args.eot_loss} | {args.attack_space}", width=80)

    def compute_one(pil_img, img_id):
        buckets = {}
        # 1. clean, undefended and defended
        core.record_dets(buckets, "clean", infer(pil_img), img_id)
        for b, img in core.build_branches(pil_img, device, args,
                                          branches).items():
            core.record_dets(buckets, f"clean+{b}", infer(img), img_id)
        # 2. OBLIVIOUS attack, then the defense
        adv_obl, obl = pgd_oblivious(pil_img, model, device, args, img_id)
        core.record_dets(buckets, OBLIVIOUS_TAG, infer(adv_obl), img_id)
        for b, img in core.build_branches(adv_obl, device, args,
                                          branches).items():
            core.record_dets(buckets, f"{OBLIVIOUS_TAG}+{b}", infer(img), img_id)
        # 3. ADAPTIVE attack (gradients taken THROUGH the defense), then defense
        adv_adp, adp = pgd_adaptive_bpda_eot(pil_img, model, device, args,
                                             branches, img_id)
        core.record_dets(buckets, ADAPTIVE_TAG, infer(adv_adp), img_id)
        for b, img in core.build_branches(adv_adp, device, args,
                                          branches).items():
            core.record_dets(buckets, f"{ADAPTIVE_TAG}+{b}", infer(img), img_id)
        # 4. the attacker's own view: J through the target, same frame
        tj = target_objectives(pil_img, model, device, args, branches,
                               {"clean": None, "oblivious": obl["adv_q"]})
        buckets[META_KEY] = {
            "target_objective": {"clean": tj["clean"],
                                 "oblivious": tj["oblivious"],
                                 "adaptive": adp["objective"]},
            "oblivious_bare_objective": obl["objective"],
            "restart_chosen": {"oblivious": obl["restart"],
                               "adaptive": adp["restart"]},
            "restart_objectives": {"oblivious": obl["restart_objectives"],
                                   "adaptive": adp["restart_objectives"]},
        }
        return buckets

    elapsed, computed, failures = core.run_per_image(
        args, "bpda", files, conditions, compute_one, ckpt,
        log_every=min(getattr(args, "log_every", 100), 25))
    ckpt.flush()

    _, cond = core.score_detection(
        args, ckpt, branches, ensembles, coco_gt,
        attack_tags=[OBLIVIOUS_TAG, ADAPTIVE_TAG])

    clean = cond["clean"]["mAP"]
    clean_def = cond[f"clean+{target}"]["mAP"]
    obl_undef, adp_undef = cond[OBLIVIOUS_TAG]["mAP"], cond[ADAPTIVE_TAG]["mAP"]
    obl_def = cond[f"{OBLIVIOUS_TAG}+{target}"]["mAP"]
    adp_def = cond[f"{ADAPTIVE_TAG}+{target}"]["mAP"]

    metas = [b[META_KEY] for b in ckpt.buckets.values() if META_KEY in b]
    tj = {k: [m["target_objective"][k] for m in metas]
          for k in ("clean", "oblivious", "adaptive")}
    below = [a < o for a, o in zip(tj["adaptive"], tj["oblivious"])]
    surrogate = {
        "definition": (f"J = sum over anchors of the max-class confidence, "
                       f"measured THROUGH {paper} with the attack's own "
                       f"surrogate (deployed forward values, {args.attack_space} "
                       f"frame, eot-loss {args.eot_loss}); lower = more "
                       f"suppression. A diagnostic of the attacker's "
                       f"optimisation, not a detection metric."),
        "images": len(metas),
        "mean_clean": _mean(tj["clean"]),
        "mean_oblivious_adv": _mean(tj["oblivious"]),
        "mean_adaptive_adv": _mean(tj["adaptive"]),
        "frac_images_adaptive_below_oblivious":
            (sum(below) / len(below)) if below else None,
    }
    uncertainty = implementation_uncertainty(args, paper, branches)
    surrogates = {b: surrogate_description(b, args) for b in branches}

    payload = {
        "phase": 4,
        "experiment": "adaptive_bpda",
        "model": adapter.model_name,
        "model_key": args.model_key,
        "purpose": ("adaptive white-box pilot: does the defense survive an "
                    "attacker who knows it is there?"),
        "smoke_test": bool(args.smoke_test),
        "target": target,
        # Every adaptive-attack flag in one place (each is also in the
        # checkpoint signature and in the blocks below).
        "attack_flags": {k: getattr(args, k) for k in ATTACK_FLAGS},
        "defense": {"name": paper, "key": target,
                    "kind": "ensemble" if is_ens else "solo",
                    "branches": branches,
                    "merge": ("class-aware NMS" if is_ens
                              else "none (solo transform)"),
                    "nms_iou": args.nms_iou},
        "attack": {"type": "PGD", "eps": args.eps, "iters": args.pgd_iters,
                   "alpha": args.alpha, "random_init": True,
                   "init": ("seeded uniform(-eps, eps) per (seed, image id, "
                            "restart); identical for the oblivious and the "
                            "adaptive attack"),
                   "init_scheme": PGD_INIT_SCHEME,
                   "seed": args.seed, "restarts": args.restarts,
                   "restart_selection": ("lowest surrogate objective on the "
                                         "quantized attack-frame tensor"),
                   "quantize": args.quantize,
                   "attack_space": args.attack_space,
                   "pixel_space": ("raw [0,1], " + (
                       f"letterboxed {args.yolo_imgsz}x{args.yolo_imgsz}"
                       if args.attack_space == "letterbox" else
                       "native resolution, differentiable ultralytics "
                       "letterbox into the model")),
                   "objective": ("minimize J = sum_anchors max_class "
                                 "pred[:,4:,:] (post-sigmoid confidences), "
                                 "i.e. maximize -J; untargeted, no ground "
                                 "truth, no self-labels, box regression "
                                 "excluded, no objectness term")},
        "bpda": {"surrogates": surrogates,
                 "median_surrogate": args.median_surrogate,
                 "identity_gradient_branches":
                     [b for b in branches if is_identity_surrogate(b, args)],
                 "true_gradient_branches":
                     [b for b in branches if not is_identity_surrogate(b, args)],
                 "forward": ("exact deployed value incl. uint8 round-trips; "
                             "round-trips straight-through in the backward"),
                 "eot_loss": args.eot_loss,
                 "eot": eot_description(args, branches)},
        "config": {k: getattr(args, k) for k in
                   ("jpeg_quality", "median_kernel", "tvm_weight", "tvm_iters",
                    "gaussian_sigma", "nms_iou")},
        "model_config": adapter.sig_fields(args),
        "implementation_uncertainty": uncertainty,
        "num_images": len(ckpt.buckets),
        "images_computed_this_run": computed,
        "select": args.select, "seed": args.seed,
        "image_dir": args.image_dir, "ann_file": args.ann_file,
        "image_ids": image_ids,
        "image_ids_sha1_16": core.ids_digest(image_ids),
        "device": str(device), "failures": failures,
        "seconds": elapsed,
        "seconds_per_image": elapsed / computed if computed else None,
        "headline": {
            "clean_mAP": clean,
            "clean_defended_mAP": clean_def,
            "oblivious_attacked_undefended_mAP": obl_undef,
            "oblivious_attacked_defended_mAP": obl_def,
            "adaptive_attacked_undefended_mAP": adp_undef,
            "adaptive_attacked_defended_mAP": adp_def,
            "defense_recovery_oblivious": obl_def - obl_undef,
            "defense_recovery_adaptive": adp_def - adp_undef,
            "adaptive_minus_oblivious_defended": adp_def - obl_def,
        },
        "surrogate_objective": surrogate,
        "conditions": cond,
    }

    md = [f"# Adaptive white-box pilot: BPDA + EOT vs {paper} "
          f"({adapter.model_name})\n"]
    md += core.header_lines(args, image_ids, device=device, extra=[
        f"- Target **{paper}** (`{target}`) = {' + '.join(branches)}"
        + (f", merged by class-aware NMS @ IoU {args.nms_iou}" if is_ens
           else ", solo transform (no merge)"),
        f"- PGD eps={args.eps}, {args.pgd_iters} iters, "
        f"alpha={args.alpha:.5f}, {args.restarts} restart(s), raw [0,1] "
        f"space; seeded init per (seed {args.seed}, image id, restart), "
        f"shared by both attacks",
        f"- Attack space `{args.attack_space}` · output quantization "
        f"`{args.quantize}` · median surrogate `{args.median_surrogate}`",
        f"- EOT loss `{args.eot_loss}`: {eot_description(args, branches)}",
        "- BPDA surrogates (backward only; every forward is the deployed "
        "value): " + "; ".join(f"{b}: {d}" for b, d in surrogates.items()),
        "- All headline numbers come from the SAME image subset.",
    ])
    md.append("## Headline (COCO mAP@[.5:.95])\n")
    md.append("| Condition | mAP |")
    md.append("|---|---|")
    md.append(f"| clean, undefended | {clean:.4f} |")
    md.append(f"| clean, {paper} | {clean_def:.4f} |")
    md.append(f"| oblivious attack, undefended | {obl_undef:.4f} |")
    md.append(f"| oblivious attack, {paper} | {obl_def:.4f} |")
    md.append(f"| adaptive attack, undefended | {adp_undef:.4f} |")
    md.append(f"| adaptive attack, {paper} | {adp_def:.4f} |")
    md.append("")
    md.append(f"- Defense recovery vs an oblivious attacker: "
              f"**{obl_def - obl_undef:+.4f}**")
    md.append(f"- Defense recovery vs an adaptive attacker: "
              f"**{adp_def - adp_undef:+.4f}**")
    md.append(f"- Defended mAP lost to adaptivity: **{adp_def - obl_def:+.4f}**\n")
    md.append("## Surrogate objective (the attacker's own view)\n")
    md.append(surrogate["definition"] + "\n")
    md.append("| Input | mean J through the target |")
    md.append("|---|---|")
    for label, key in (("clean", "mean_clean"),
                       ("oblivious adversarial", "mean_oblivious_adv"),
                       ("adaptive adversarial", "mean_adaptive_adv")):
        md.append(f"| {label} | {core.fmt(surrogate[key])} |")
    md.append("")
    md.append(f"- Images where the adaptive J is below the oblivious J: "
              f"**{core.fmt(surrogate['frac_images_adaptive_below_oblivious'], 3)}**"
              f" of {surrogate['images']} (should be close to 1; if not, the "
              f"adaptive attack is not optimising its objective)\n")
    md.append("## All conditions\n")
    md.append("| Input | Defense | Tag | mAP | AP50 |")
    md.append("|---|---|---|---|---|")
    for tag in sorted(cond):
        inp, dfn = condition_label(tag)
        md.append(f"| {inp} | {dfn} | `{tag}` | {cond[tag]['mAP']:.4f} | "
                  f"{cond[tag]['AP50']:.4f} |")
    md.append("")
    md.append("## Implementation uncertainty\n")
    md += [f"- {u}" for u in uncertainty]
    md.append("")
    md += core.timing_lines(elapsed, computed, len(files), str(device),
                            targets=(300, 1000, 5000))
    md_text = "\n".join(md)

    stem = summary_stem(args)
    core.write_per_image(args, stem, {
        "note": ("per-image detection COUNTS per condition, plus the "
                 "attacker's surrogate objectives and chosen restarts"),
        "counts": {str(i): {c: len(b.get(c, [])) for c in conditions}
                   for i, b in ckpt.buckets.items()},
        "attack_meta": {str(i): b.get(META_KEY)
                        for i, b in ckpt.buckets.items()}})
    core.write_summary(args, payload, stem)
    core.write_markdown(args, md_text, stem)
    # Headline only: the full table and the uncertainty list are in the .md
    # written above, so repeating them would double them inside run.log.
    print("\n" + "=" * 70)
    print(f"RESULT DIGEST: BPDA+EOT vs {paper} (full table in the .md)")
    print("=" * 70)
    print(f"  clean, undefended                : {clean:.4f}")
    print(f"  clean, {paper:<26}: {clean_def:.4f}")
    print(f"  oblivious attack, undefended     : {obl_undef:.4f}")
    print(f"  oblivious attack, {paper:<14}: {obl_def:.4f}  "
          f"(recovery {obl_def - obl_undef:+.4f})")
    print(f"  adaptive  attack, undefended     : {adp_undef:.4f}")
    print(f"  adaptive  attack, {paper:<14}: {adp_def:.4f}  "
          f"(recovery {adp_def - adp_undef:+.4f})")
    print(f"  defended mAP lost to adaptivity  : {adp_def - obl_def:+.4f}")
    print(f"  surrogate J clean/obl/adp        : "
          f"{core.fmt(surrogate['mean_clean'], 2)} / "
          f"{core.fmt(surrogate['mean_oblivious_adv'], 2)} / "
          f"{core.fmt(surrogate['mean_adaptive_adv'], 2)}  (adaptive below "
          f"oblivious on "
          f"{core.fmt(surrogate['frac_images_adaptive_below_oblivious'], 3)})")
    core.print_runtime_estimate(elapsed, computed, str(device))
    return payload


# ======================================================================
# 8. Entry point.
# ======================================================================
def _positive_int(text):
    import argparse
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
    return v


def parse_args(argv=None):
    p = core.base_parser("Phase-4 YOLOv8x-worldv2 runs.", EXPERIMENTS)
    g = p.add_argument_group("attack (adaptive_bpda only; Phase-3 values)")
    g.add_argument("--eps", type=float, default=0.03)
    g.add_argument("--pgd-iters", type=int, default=None,
                   help="PGD iterations (default 10; 2 under --smoke-test, "
                        "where ~40 CPU forward/backward passes per image would "
                        "otherwise take tens of minutes).")
    g.add_argument("--alpha", type=float, default=None,
                   help="PGD step size (default eps/4, the Phase-3 value).")
    a = p.add_argument_group("adaptive attack (adaptive_bpda only)")
    a.add_argument("--target", choices=list(ADAPTIVE_TARGETS),
                   default=DEFAULT_TARGET,
                   help="Defense the adaptive attacker differentiates through "
                        "and that is scored. Ensembles (EnsJMG default, Ens4W, "
                        "EnsJMT, EnsBTC) take their members from "
                        "phase3_common; solos: median, jpeg, gaussian, tvm, "
                        "blur_tvm.")
    a.add_argument("--eot-loss", choices=["mean", "max"], default="max",
                   help="max (default) = union-aware: per anchor, suppress the "
                        "most confident branch (the NMS-union keeps a box if "
                        "ANY branch keeps it). mean = average of the branch "
                        "objectives (the earlier pilot).")
    a.add_argument("--restarts", type=_positive_int, default=1,
                   help="Seeded PGD restarts per image (both attacks); keeps "
                        "the one with the lowest surrogate objective.")
    a.add_argument("--median-surrogate", choices=["true", "identity"],
                   default="true",
                   help="Median backward: true a.e. gradient of the unfold "
                        "median (default) or BPDA identity.")
    a.add_argument("--quantize", choices=["truncate", "round"],
                   default="truncate",
                   help="Final uint8 conversion of the attack output. "
                        "truncate = Phase-3 compatible.")
    a.add_argument("--attack-space", choices=["letterbox", "native"],
                   default="letterbox",
                   help="letterbox (Phase-3 compatible) = optimise the square "
                        "640 letterbox. native = optimise at native resolution "
                        "where the defense runs, through a differentiable copy "
                        "of ultralytics' predict letterbox.")
    m = p.add_argument_group("model (Phase-3 values)")
    m.add_argument("--yolo-model", default="yolov8x-worldv2.pt")
    m.add_argument("--yolo-imgsz", type=int, default=640)
    m.add_argument("--yolo-conf", type=float, default=0.001)
    m.add_argument("--yolo-iou", type=float, default=0.5)

    args = p.parse_args(argv)
    core.finalize_args(args, "yolo")
    if args.alpha is None:
        args.alpha = args.eps / 4.0
    if args.pgd_iters is None:
        args.pgd_iters = 2 if args.smoke_test else 10
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.preflight:
        return core.preflight(
            args,
            modules=[("numpy", "all"), ("PIL", "all"), ("torch", "all"),
                     ("ultralytics", "yolo inference"),
                     ("pycocotools", "COCO mAP")],
            weights=[(args.yolo_model, os.path.isfile(args.yolo_model))])

    device = core.load_runtime(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with core.run_log(os.path.join(args.output_dir, "run.log"), argv=sys.argv,
                      extra={"experiment": args.experiment, "n": args.n,
                             "device": str(device),
                             "smoke_test": args.smoke_test}):
        args.image_dir = core.resolve_image_dir(args.image_dir)
        args.ann_file = core.resolve_ann_file(args.ann_file)
        files, image_ids = core.select_images(args.image_dir, args.n, args.seed,
                                             args.select,
                                             allow_short=args.smoke_test)
        args.n = len(files)

        rt.pc.print_banner(f"PHASE 4: YOLO: {args.experiment}", width=80)
        if args.smoke_test:
            print(f"*** SMOKE TEST: {args.n} images on {device}. The numbers "
                  f"below are NOT results. ***")
        print(f"  run id    : {args.run_id}")
        print(f"  images    : {args.n} from {args.image_dir} "
              f"(--select {args.select}, seed {args.seed})")
        print(f"  id digest : {core.ids_digest(image_ids)}")
        print(f"  device    : {device}")
        print(f"  output    : {args.output_dir}")

        from pycocotools.coco import COCO
        coco_gt = COCO(args.ann_file)
        adapter = YoloAdapter(args, device)
        adapter.load(coco_gt)

        if args.experiment == "clean_control":
            core.clean_control(adapter, args, coco_gt, files, image_ids)
        elif args.experiment == "adaptive_bpda":
            adaptive_bpda(adapter, args, coco_gt, files, image_ids)
        else:  # pragma: no cover (argparse restricts the choices)
            raise ValueError(f"unknown experiment {args.experiment}")

        core.finish(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
