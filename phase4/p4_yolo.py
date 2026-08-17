#!/usr/bin/env python3
# ======================================================================
# p4_yolo.py — the YOLOv8x-worldv2 entry point for Phase 4.
#
#   python phase4/p4_yolo.py --experiment clean_control  --n 1000 --gpu 0
#   python phase4/p4_yolo.py --experiment adaptive_bpda  --n 300  --gpu 0
#   python phase4/p4_yolo.py --preflight
#   python phase4/p4_yolo.py --experiment adaptive_bpda --smoke-test
#
# One process, one model, one GPU — designed to be one tmux pane. Everything
# that is not YOLO-specific (CLI, image selection, defense branches, ensemble
# NMS merge, checkpointing, COCO scoring, reports, artifact policy) comes from
# p4_core.py, so it is written once and shared with p4_florence.py.
#
# THIS FILE CONTAINS: the letterbox preprocessing, the detection parsing, the
# attack objective, the oblivious PGD, and the adaptive BPDA+EOT PGD.
# ======================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p4_core as core                                     # noqa: E402
from p4_core import rt                                     # noqa: E402

EXPERIMENTS = ("clean_control", "adaptive_bpda")
ENSEMBLE_KEY = "ens_jpeg_median_gaussian"                  # EnsJMG
ENSEMBLE_PAPER_NAME = "EnsJMG"
OBLIVIOUS_TAG = "pgd_oblivious"
ADAPTIVE_TAG = "pgd_adaptive"


# ======================================================================
# 1. Letterbox + detection parsing — VERBATIM from Phase 3
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
        print(f"[yolo] loaded {self.args.yolo_model} — {len(names)} COCO "
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
# YOLO EMITS NO TOKENS — there is no cross-entropy here, and the paper must not
# describe it as one. `pred = model.model(x)[0]` is [1, 4+80, 8400]; the class
# block `pred[:, 4:, :]` holds POST-SIGMOID class confidences (ultralytics'
# WorldDetect head concatenates `cls.sigmoid()` onto the decoded boxes). The
# objective takes the max over the 80 classes per anchor and sums over all 8400
# anchors, so ascending it suppresses total peak confidence. Untargeted; no
# ground truth, no self-labels, and the box block pred[:, :4, :] is excluded.
# YOLOv8 is anchor-free, so there is no objectness term to include.
# VERBATIM objective from archive/Phase3/run_survey_yolo.py:243-247.
# ======================================================================
def yolo_confidence_loss(model, x):
    # The Detect head caches anchor tensors keyed on input shape; they are not
    # part of a new autograd graph, so the cache must be reset before EVERY
    # autograd forward (archive/Phase3/run_survey_yolo.py:243/266/303).
    model.model.model[-1].shape = None
    preds = model.model(x)
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    cls_scores = pred[:, 4:, :]
    return -cls_scores.max(dim=1)[0].sum()


def pgd_oblivious(pil_img, model, device, args):
    """PGD on the bare model — VERBATIM from run_survey_yolo.py:255-280.

    The attacker does not know a defense exists.
    """
    torch, np = rt.torch, rt.np
    orig_size = pil_img.size
    lb, _, pl, pt, nw, nh = letterbox(pil_img, args.yolo_imgsz)
    img_np = np.array(lb).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    adv = (img_t.clone().detach()
           + torch.empty_like(img_t).uniform_(-args.eps, args.eps)).clamp(0.0, 1.0)
    for _ in range(args.pgd_iters):
        adv = adv.detach().requires_grad_(True)
        loss = yolo_confidence_loss(model, adv)
        if adv.grad is not None:
            adv.grad.zero_()
        loss.backward()
        with torch.no_grad():
            adv = adv + args.alpha * adv.grad.sign()
            delta = torch.clamp(adv - img_t, min=-args.eps, max=args.eps)
            adv = torch.clamp(img_t + delta, 0.0, 1.0).detach()

    adv_np = (adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return unletterbox(adv_np, orig_size, pl, pt, nw, nh)


# ======================================================================
# 3. BPDA surrogates (Athalye, Carlini & Wagner, ICML 2018).
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


def assert_gaussian_matches(device, sigma, tol=1e-6):
    """Refuse to run if the BPDA Gaussian differs from the deployed one."""
    torch = rt.torch
    g = torch.Generator(device="cpu").manual_seed(0)
    probe = torch.rand(1, 3, 64, 64, generator=g).to(device)
    diff = (rt.pc.gaussian_gpu(probe, sigma=sigma)
            - gaussian_differentiable(probe, sigma)).abs().max().item()
    assert diff < tol, (
        f"gaussian_differentiable deviates from pc.gaussian_gpu by {diff:.2e} "
        f"(> {tol:.0e}); the BPDA gradient would not match the deployed "
        f"defense. Refusing to run.")
    print(f"[guard] gaussian_differentiable == pc.gaussian_gpu "
          f"(max |diff| = {diff:.2e})")


def _quantize_uint8(t):
    """The PIL uint8 round-trip the deployed pipeline performs."""
    return rt.pc.pil_to_tensor(rt.pc.tensor_to_pil(t), t.device)


def bpda_branch(adv, branch, args):
    """Forward = the REAL transform. Backward = BPDA.

    jpeg / median : non-differentiable -> straight-through identity gradient
                    via `adv + (value - adv).detach()`, which forwards `value`
                    while d(out)/d(adv) = I.
    gaussian      : differentiable -> its TRUE gradient, no straight-through.
    """
    if branch == "gaussian":
        return gaussian_differentiable(adv, args.gaussian_sigma)
    with rt.torch.no_grad():
        if branch == "jpeg":
            value = rt.pc.pil_to_tensor(
                rt.pc.jpeg_cpu(rt.pc.tensor_to_pil(adv.detach()),
                               args.jpeg_quality), adv.device)
        elif branch == "median":
            value = _quantize_uint8(
                rt.pc.median_gpu(_quantize_uint8(adv.detach()),
                                 args.median_kernel))
        else:
            raise ValueError(f"unknown BPDA branch '{branch}'")
    return adv + (value - adv).detach()


def pgd_adaptive_bpda_eot(pil_img, model, device, args, branches):
    """PGD whose per-step gradient is the EOT mean over the defense branches."""
    torch, np = rt.torch, rt.np
    orig_size = pil_img.size
    lb, _, pl, pt, nw, nh = letterbox(pil_img, args.yolo_imgsz)
    img_np = np.array(lb).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Same random init as the oblivious attack, so the two differ only in the
    # gradient they follow.
    adv = (img_t.clone().detach()
           + torch.empty_like(img_t).uniform_(-args.eps, args.eps)).clamp(0.0, 1.0)
    for _ in range(args.pgd_iters):
        adv = adv.detach().requires_grad_(True)
        grads = []
        for branch in branches:
            defended = bpda_branch(adv, branch, args)
            loss = yolo_confidence_loss(model, defended)
            (g,) = torch.autograd.grad(loss, adv)
            grads.append(g.detach())
        grad = torch.stack(grads, dim=0).mean(dim=0)     # EOT over the ensemble
        with torch.no_grad():
            adv = adv + args.alpha * grad.sign()
            delta = torch.clamp(adv - img_t, min=-args.eps, max=args.eps)
            adv = torch.clamp(img_t + delta, 0.0, 1.0).detach()

    adv_np = (adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return unletterbox(adv_np, orig_size, pl, pt, nw, nh)


IMPLEMENTATION_UNCERTAINTY = [
    "The attack optimizes in 640x640 letterbox space while the deployed "
    "defense runs on the full-resolution un-letterboxed image, so the BPDA "
    "surrogate applies the transforms at a different scale — the adaptive "
    "damage reported here is a LOWER bound on a fully matched attack.",
    "The adversarial tensor is bicubic-resampled back to native resolution "
    "outside the gradient path (as in every Phase-3 attack), which partially "
    "destroys the perturbation.",
    "BPDA's identity Jacobian for JPEG/median is the standard approximation, "
    "not those transforms' true Jacobian; a learned differentiable proxy would "
    "give a stronger attacker.",
    "The median surrogate reproduces the deployed uint8 round-trip; the "
    "Gaussian surrogate does not quantize, because that would destroy the true "
    "gradient it exists to provide.",
    "EOT here averages over the 3 deterministic defense branches, not over "
    "defense randomness — EnsJMG has none. A randomized defense would need "
    "EOT samples per branch as well.",
]


# ======================================================================
# 4. The adaptive white-box experiment (YOLO only — no beam search, so it is
#    cheap enough to run a real 10-step PGD three times per image).
# ======================================================================
def adaptive_bpda(adapter, args, coco_gt, files, image_ids):
    ensembles = core.load_ensembles([ENSEMBLE_KEY])
    branches = ensembles[ENSEMBLE_KEY]
    assert sorted(branches) == ["gaussian", "jpeg", "median"], (
        f"{ENSEMBLE_KEY} members changed to {branches}; this pilot's BPDA "
        f"surrogates only cover jpeg/median/gaussian.")
    device, model = adapter.device, adapter.model
    infer = adapter.detection_fn("yolo")
    assert_gaussian_matches(device, args.gaussian_sigma)

    conditions = (["clean"] + [f"clean+{b}" for b in branches]
                  + [OBLIVIOUS_TAG] + [f"{OBLIVIOUS_TAG}+{b}" for b in branches]
                  + [ADAPTIVE_TAG] + [f"{ADAPTIVE_TAG}+{b}" for b in branches])

    sig = rt.pc.config_signature(
        phase=4, experiment="adaptive_bpda", model=adapter.model_name,
        branches=sorted(branches), ensemble=ENSEMBLE_KEY, eps=args.eps,
        iters=args.pgd_iters, alpha=args.alpha, jpeg_quality=args.jpeg_quality,
        median_kernel=args.median_kernel, gaussian_sigma=args.gaussian_sigma,
        nms_iou=args.nms_iou, n=len(files), ids=core.ids_digest(image_ids),
        device=str(device), **adapter.sig_fields(args))
    ckpt = core.make_checkpoint(args, "adaptive", sig)

    rt.pc.print_banner(
        f"ADAPTIVE WHITE-BOX — BPDA+EOT vs {ENSEMBLE_PAPER_NAME} | "
        f"n={len(files)} | PGD eps={args.eps} x{args.pgd_iters}", width=80)

    def compute_one(pil_img, img_id):
        buckets = {}
        # 1. clean, undefended and defended
        core.record_dets(buckets, "clean", infer(pil_img), img_id)
        for b, img in core.build_branches(pil_img, device, args,
                                          branches).items():
            core.record_dets(buckets, f"clean+{b}", infer(img), img_id)
        # 2. OBLIVIOUS attack, then the defense
        adv_obl = pgd_oblivious(pil_img, model, device, args)
        core.record_dets(buckets, OBLIVIOUS_TAG, infer(adv_obl), img_id)
        for b, img in core.build_branches(adv_obl, device, args,
                                          branches).items():
            core.record_dets(buckets, f"{OBLIVIOUS_TAG}+{b}", infer(img), img_id)
        # 3. ADAPTIVE attack (gradients taken THROUGH the defense), then defense
        adv_adp = pgd_adaptive_bpda_eot(pil_img, model, device, args, branches)
        core.record_dets(buckets, ADAPTIVE_TAG, infer(adv_adp), img_id)
        for b, img in core.build_branches(adv_adp, device, args,
                                          branches).items():
            core.record_dets(buckets, f"{ADAPTIVE_TAG}+{b}", infer(img), img_id)
        return buckets

    elapsed, computed, failures = core.run_per_image(
        args, "bpda", files, conditions, compute_one, ckpt, log_every=10)
    ckpt.flush()

    _, cond = core.score_detection(
        args, ckpt, branches, ensembles, coco_gt,
        attack_tags=[OBLIVIOUS_TAG, ADAPTIVE_TAG])

    clean = cond["clean"]["mAP"]
    obl_undef, adp_undef = cond[OBLIVIOUS_TAG]["mAP"], cond[ADAPTIVE_TAG]["mAP"]
    obl_ens = cond[f"{OBLIVIOUS_TAG}+{ENSEMBLE_KEY}"]["mAP"]
    adp_ens = cond[f"{ADAPTIVE_TAG}+{ENSEMBLE_KEY}"]["mAP"]

    payload = {
        "phase": 4,
        "experiment": "adaptive_bpda",
        "model": adapter.model_name,
        "model_key": args.model_key,
        "purpose": ("adaptive white-box pilot: does the defense survive an "
                    "attacker who knows it is there?"),
        "smoke_test": bool(args.smoke_test),
        "defense": {"name": ENSEMBLE_PAPER_NAME, "key": ENSEMBLE_KEY,
                    "branches": branches, "merge": "class-aware NMS",
                    "nms_iou": args.nms_iou},
        "attack": {"type": "PGD", "eps": args.eps, "iters": args.pgd_iters,
                   "alpha": args.alpha, "random_init": True,
                   "pixel_space": "raw [0,1] (letterboxed 640x640)",
                   "objective": ("maximize -sum_anchors max_class "
                                 "pred[:,4:,:] (post-sigmoid confidences); "
                                 "untargeted, no ground truth, no self-labels, "
                                 "box regression excluded, no objectness term")},
        "bpda": {"identity_gradient_branches": ["jpeg", "median"],
                 "true_gradient_branches": ["gaussian"],
                 "eot": "per-step gradient = mean over the 3 branch gradients"},
        "implementation_uncertainty": IMPLEMENTATION_UNCERTAINTY,
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
            "oblivious_attacked_undefended_mAP": obl_undef,
            "oblivious_attacked_defended_mAP": obl_ens,
            "adaptive_attacked_undefended_mAP": adp_undef,
            "adaptive_attacked_defended_mAP": adp_ens,
            "defense_recovery_oblivious": obl_ens - obl_undef,
            "defense_recovery_adaptive": adp_ens - adp_undef,
            "adaptive_minus_oblivious_defended": adp_ens - obl_ens,
        },
        "conditions": cond,
    }

    md = [f"# Adaptive white-box pilot — BPDA + EOT vs {ENSEMBLE_PAPER_NAME} "
          f"({adapter.model_name})\n"]
    md += core.header_lines(args, image_ids, extra=[
        f"- PGD eps={args.eps}, {args.pgd_iters} iters, alpha={args.alpha:.5f}, "
        f"random init, raw [0,1] space",
        f"- Defense {ENSEMBLE_PAPER_NAME} = {' + '.join(branches)}, merged by "
        f"class-aware NMS @ IoU {args.nms_iou}",
        "- BPDA: identity gradient for JPEG/median, true gradient for Gaussian. "
        "EOT: mean of the 3 branch gradients per step.",
        "- All three headline numbers come from the SAME image subset.",
    ])
    md.append("## Headline (COCO mAP@[.5:.95])\n")
    md.append("| Condition | mAP |")
    md.append("|---|---|")
    md.append(f"| clean, undefended | {clean:.4f} |")
    md.append(f"| oblivious attack, undefended | {obl_undef:.4f} |")
    md.append(f"| oblivious attack, {ENSEMBLE_PAPER_NAME} | {obl_ens:.4f} |")
    md.append(f"| adaptive attack, undefended | {adp_undef:.4f} |")
    md.append(f"| adaptive attack, {ENSEMBLE_PAPER_NAME} | {adp_ens:.4f} |")
    md.append("")
    md.append(f"- Defense recovery vs an oblivious attacker: "
              f"**{obl_ens - obl_undef:+.4f}**")
    md.append(f"- Defense recovery vs an adaptive attacker: "
              f"**{adp_ens - adp_undef:+.4f}**")
    md.append(f"- Defended mAP lost to adaptivity: **{adp_ens - obl_ens:+.4f}**\n")
    md.append("## All conditions\n")
    md.append("| Condition | mAP | AP50 |")
    md.append("|---|---|---|")
    for tag in sorted(cond):
        md.append(f"| `{tag}` | {cond[tag]['mAP']:.4f} | {cond[tag]['AP50']:.4f} |")
    md.append("")
    md.append("## Implementation uncertainty\n")
    md += [f"- {u}" for u in IMPLEMENTATION_UNCERTAINTY]
    md.append("")
    md += core.timing_lines(elapsed, computed, len(files), str(device),
                            targets=(300, 1000, 5000))
    md_text = "\n".join(md)

    core.write_per_image(args, "adaptive_bpda", {
        "note": "per-image detection COUNTS per condition",
        "counts": {str(i): {c: len(b.get(c, [])) for c in conditions}
                   for i, b in ckpt.buckets.items()}})
    stem = f"adaptive_bpda_{args.model_key}"
    core.write_summary(args, payload, stem)
    core.write_markdown(args, md_text, stem)
    print("\n" + md_text)
    core.print_runtime_estimate(elapsed, computed, str(device))
    return payload


# ======================================================================
# 5. Entry point.
# ======================================================================
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
                                             args.select)
        args.n = len(files)

        rt.pc.print_banner(f"PHASE 4 — YOLO — {args.experiment}", width=80)
        if args.smoke_test:
            print("*** SMOKE TEST: 2 images on CPU. The numbers below are NOT "
                  "results. ***")
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
        else:  # pragma: no cover — argparse restricts the choices
            raise ValueError(f"unknown experiment {args.experiment}")

        core.finish(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
