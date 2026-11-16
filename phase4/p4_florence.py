#!/usr/bin/env python3
# ======================================================================
# p4_florence.py — the Florence-2-base entry point for Phase 4.
#
#   python phase4/p4_florence.py --experiment clean_control --n 1000 --gpu 0
#   python phase4/p4_florence.py --experiment clean_control --tracks florence_ocr
#   python phase4/p4_florence.py --preflight
#   python phase4/p4_florence.py --experiment clean_control --smoke-test
#
# One process, one model, one GPU — one tmux pane. The model is loaded ONCE and
# serves both tracks (`<OD>` detection and `<OCR>`), because loading Florence-2
# twice is the single most wasteful thing this pipeline can do. Everything that
# is not Florence-specific comes from p4_core.py.
#
# TWO THINGS THE PAPER MUST KEEP DISCLOSING
# 1. Florence `<OD>` emits no confidences, so `_compute_score` FABRICATES a
#    geometric pseudo-score (area ratio + centeredness). It is applied
#    identically to every condition, so DELTAS are fair, but the absolute
#    Florence mAP is geometry-ranked, not confidence-ranked.
# 2. The OCR metric is SELF-CONSISTENCY against the model's own clean output,
#    not accuracy — COCO has no OCR ground truth.
#
# ATTACKS IN THIS FILE: only the adaptive experiment (`adaptive_bpda`, section
# 2): oblivious PGD and PGD through the defense branches on the `<OD>` track,
# in PIXEL space at native resolution, sharing p4_yolo's PGD loop, seeded
# init, quantization and BPDA surrogates. The Phase-3 FGSM/PGD/Patch attacks
# (normalized space) stay in archive/Phase3/ and are not ported.
# ======================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p4_core as core                                     # noqa: E402
import p4_yolo as ya              # noqa: E402  (shared attack machinery only)
from p4_core import rt                                     # noqa: E402

EXPERIMENTS = ("clean_control", "adaptive_bpda")
# Florence always attacks at native resolution, so `native` is its default
# and never appears in a summary name.
STEM_FLAG_DEFAULTS = dict(ya.STEM_FLAG_DEFAULTS, attack_space="native")
ALL_TRACKS = ("florence_det", "florence_ocr")


# ======================================================================
# 1. Florence -> COCO label mapping. VERBATIM from Phase 3
#    (archive/Phase3/run_survey_florence_detection.py:259-347).
# ======================================================================
FLORENCE_TO_COCO = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "baby": "person", "kid": "person", "player": "person",
    "pedestrian": "person", "human": "person", "skier": "person",
    "snowboarder": "person", "surfer": "person", "rider": "person",
    "automobile": "car", "van": "car", "sedan": "car", "suv": "car",
    "taxi": "car", "minivan": "car",
    "motor bike": "motorcycle", "motorbike": "motorcycle",
    "aeroplane": "airplane", "aircraft": "airplane", "jet": "airplane",
    "lorry": "truck", "pickup truck": "truck",
    "television": "tv", "tv set": "tv", "monitor": "tv", "screen": "tv",
    "television set": "tv",
    "mobile phone": "cell phone", "cellphone": "cell phone",
    "smartphone": "cell phone", "phone": "cell phone",
    "computer keyboard": "keyboard", "computer mouse": "mouse",
    "notebook computer": "laptop", "notebook": "laptop",
    "studio couch": "couch", "sofa": "couch", "settee": "couch",
    "kitchen & dining room table": "dining table", "table": "dining table",
    "desk": "dining table",
    "swivel chair": "chair", "armchair": "chair", "stool": "chair",
    "puppy": "dog", "kitten": "cat",
    "ski": "skis", "ski pole": "skis",
    "racket": "tennis racket",
    "ball": "sports ball", "football": "sports ball",
    "soccer ball": "sports ball", "baseball": "sports ball",
    "basketball": "sports ball", "tennis ball": "sports ball",
    "glove": "baseball glove",
    "houseplant": "potted plant", "plant": "potted plant",
    "flower pot": "potted plant", "flowerpot": "potted plant",
    "wine bottle": "bottle", "beer bottle": "bottle", "water bottle": "bottle",
    "drinking glass": "wine glass", "glass": "wine glass", "goblet": "wine glass",
    "pocketknife": "knife", "kitchen knife": "knife", "butter knife": "knife",
    "hair dryer": "hair drier", "hairdryer": "hair drier",
    "blow dryer": "hair drier",
    "wristwatch": "clock", "wall clock": "clock", "alarm clock": "clock",
    "bag": "handbag", "purse": "handbag", "wristlet": "handbag",
    "briefcase": "suitcase", "luggage": "suitcase", "travel bag": "suitcase",
    "backpack bag": "backpack",
    "traffic signal": "traffic light",
    "fire plug": "fire hydrant",
}


def compute_score(box, img_w, img_h):
    """FABRICATED geometric pseudo-confidence. VERBATIM from Phase 3
    (run_survey_florence_detection.py:312-322).

    Florence `<OD>` returns no scores, and COCOeval needs a ranking. This uses
    area ratio + centeredness. Applied identically to every condition, so the
    deltas this repo reports are fair; the absolute mAP is geometry-ranked.
    Must stay disclosed in the paper.
    """
    np = rt.np
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    img_area = img_w * img_h
    box_area = w * h
    area_ratio = min(box_area / img_area, 0.5) if img_area > 0 else 0
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    icx, icy = img_w / 2, img_h / 2
    cd = np.sqrt(((cx - icx) / img_w) ** 2 + ((cy - icy) / img_h) ** 2)
    s = 0.6 + 0.2 * area_ratio + 0.15 * (1 - cd)
    return min(0.98, max(0.6, s))


def _box_iou_local(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    u = aa + ab - inter
    return inter / u if u > 0 else 0


def non_max_suppression(boxes, labels, scores, iou_thr=0.5):
    """VERBATIM from run_survey_florence_detection.py:335-347."""
    np = rt.np
    if not boxes:
        return [], [], []
    boxes = np.array(boxes)
    idxs = np.argsort(scores)[::-1]
    keep, kl, ks = [], [], []
    for i in idxs:
        sup = False
        for j in keep:
            if labels[i] == labels[j] and _box_iou_local(boxes[i], boxes[j]) > iou_thr:
                sup = True
                break
        if not sup:
            keep.append(i); kl.append(labels[i]); ks.append(scores[i])
    return boxes[keep].tolist(), kl, ks


class FlorenceAdapter:
    """Loads Florence-2-base once; serves both the `<OD>` and `<OCR>` tracks."""

    def __init__(self, args, device, tracks):
        self.args = args
        self.device = device
        self.tracks = tuple(tracks)
        self.model_name = args.florence_model
        self.model = None
        self.processor = None
        self.torch_dtype = None
        self.category_mapping = {}

    def sig_fields(self, args):
        return {"florence_model": args.florence_model,
                "revision": args.florence_revision,
                "num_beams": args.num_beams,
                "det_max_new_tokens": args.det_max_new_tokens,
                "ocr_max_new_tokens": args.ocr_max_new_tokens,
                "repetition_penalty": args.repetition_penalty,
                "length_penalty": args.length_penalty}

    def load(self, coco_gt):
        """Load with the Phase-3 flash-attn / erfinv workarounds.

        float16 on GPU (as Phase 3 did); float32 on CPU, where half precision
        is unsupported for several ops — the dtype is recorded in the summary so
        a CPU smoke run is never mistaken for a comparable measurement.
        """
        from unittest.mock import patch
        from transformers import AutoProcessor, AutoModelForCausalLM
        from transformers.dynamic_module_utils import get_imports
        torch = rt.torch
        a = self.args
        self.torch_dtype = (torch.float32 if self.device.type == "cpu"
                            else torch.float16)

        # Florence-2's modeling file imports flash_attn unconditionally, but
        # every use is guarded; drop it from the static dependency check so the
        # model loads with eager/SDPA attention where flash-attn is absent.
        def _imports_without_flash_attn(filename):
            return [i for i in get_imports(filename) if i != "flash_attn"]

        self.processor = AutoProcessor.from_pretrained(
            a.florence_model, revision=a.florence_revision,
            trust_remote_code=True)
        with patch("transformers.dynamic_module_utils.get_imports",
                   _imports_without_flash_attn):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    a.florence_model, revision=a.florence_revision,
                    torch_dtype=self.torch_dtype, trust_remote_code=True,
                ).to(self.device).eval()
            except RuntimeError as exc:
                # Older torch cannot run float16 weight init on CPU
                # ("erfinv_vml_cpu not implemented for 'Half'").
                if "erfinv" not in str(exc):
                    raise
                self.model = AutoModelForCausalLM.from_pretrained(
                    a.florence_model, revision=a.florence_revision,
                    torch_dtype=torch.float32, trust_remote_code=True,
                ).to(self.device).to(self.torch_dtype).eval()
        if coco_gt is not None:
            self.category_mapping = {
                c["name"]: c["id"]
                for c in coco_gt.loadCats(coco_gt.getCatIds())}
        print(f"[florence] loaded {a.florence_model} @ {a.florence_revision} "
              f"| dtype={self.torch_dtype} | device={self.device} "
              f"| tracks={list(self.tracks)}")

    def _map_label(self, label):
        cm = self.category_mapping
        if label in cm:
            return label
        m = FLORENCE_TO_COCO.get(label)
        if m and m in cm:
            return m
        lo = label.lower()
        if lo in cm:
            return lo
        m2 = FLORENCE_TO_COCO.get(lo)
        if m2 and m2 in cm:
            return m2
        return None

    def detection_fn(self, track="florence_det"):
        """run_inference VERBATIM from run_survey_florence_detection.py:350-378."""
        a, torch = self.args, rt.torch
        assert self.category_mapping, "COCO categories were not loaded"

        def run_inference(pil_img):
            img_w, img_h = pil_img.size
            with torch.no_grad():
                inputs = self.processor(text="<OD>", images=pil_img,
                                        return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device)
                pixel_values = inputs.pixel_values.to(device=self.device,
                                                      dtype=self.torch_dtype)
                gen_ids = self.model.generate(
                    input_ids=input_ids, pixel_values=pixel_values,
                    max_new_tokens=a.det_max_new_tokens,
                    num_beams=a.num_beams, do_sample=False,
                    repetition_penalty=a.repetition_penalty,
                    length_penalty=a.length_penalty)
                txt = self.processor.batch_decode(
                    gen_ids, skip_special_tokens=False)[0]
                parsed = self.processor.post_process_generation(
                    txt, task="<OD>", image_size=(img_w, img_h)) or {}
            od = parsed.get("<OD>", {})
            bboxes, labels = od.get("bboxes", []), od.get("labels", [])
            scores = [compute_score(b, img_w, img_h) for b in bboxes]
            kb, kl, ks = non_max_suppression(bboxes, labels, scores,
                                             iou_thr=a.nms_iou)
            results = []
            for box, lab, sc in zip(kb, kl, ks):
                mp = self._map_label(lab)
                if mp is None:
                    continue
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                results.append({"bbox": [x1, y1, w, h],
                                "category_id": self.category_mapping[mp],
                                "score": sc})
            return results

        return run_inference

    def ocr_fn(self):
        """get_ocr_text VERBATIM from run_survey_florence_ocr.py:211-242."""
        a, torch = self.args, rt.torch

        def get_ocr_text(pil_img):
            img_w, img_h = pil_img.size
            with torch.inference_mode():
                inputs = self.processor(text="<OCR>", images=pil_img,
                                        return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device)
                pixel_values = inputs.pixel_values.to(device=self.device,
                                                      dtype=self.torch_dtype)
                gen_ids = self.model.generate(
                    input_ids=input_ids, pixel_values=pixel_values,
                    max_new_tokens=a.ocr_max_new_tokens,
                    num_beams=a.num_beams, do_sample=False,
                    repetition_penalty=a.repetition_penalty,
                    length_penalty=a.length_penalty)
                raw = self.processor.batch_decode(
                    gen_ids, skip_special_tokens=False)[0]
                try:
                    parsed = self.processor.post_process_generation(
                        raw, task="<OCR>", image_size=(img_w, img_h))
                    return core.decode_ocr(parsed)
                except Exception:  # noqa: BLE001
                    return core.sanitize_generated_text(raw)

        return get_ocr_text


# ======================================================================
# 2. The adaptive white-box experiment on the <OD> track.
#
# OBJECTIVE. Florence-2 is attacked as in Phase 3, with self-labelled
# token-level cross-entropy: the clean image is decoded once (the scored
# decode settings), the resulting ids are the labels, and the attack
# MAXIMIZES the teacher-forced cross-entropy of the perturbed image against
# them. The shared PGD loop minimizes, so the objective here is J = -CE.
#
# WHAT DIFFERS FROM PHASE 3. The perturbation lives in PIXEL space [0,1] at
# the image's native resolution (Phase 3 perturbed the normalized 768x768
# tensor, where eps=0.03 is only ~1.7/255). So eps=0.03 here is the same
# 7/255 budget as the YOLO experiment, every defense branch sees the image
# at the scale at which it is deployed, and nothing resamples the result.
# These numbers are therefore NOT comparable with the Phase-3 Florence runs.
#
# ENSEMBLES (--eot-loss). The NMS merge keeps a box if ANY branch keeps it,
# so the attacker has to break every branch. `max` (default) raises the
# cross-entropy of the LEAST damaged branch at each step, J = max_b(-CE_b);
# `mean` averages the branch losses. Same flag and meaning as p4_yolo.
# ======================================================================
def processor_geometry(processor):
    """(height, width, mean, std) of the processor's image preprocessing."""
    ip = getattr(processor, "image_processor", processor)
    size = getattr(ip, "size", None) or {}
    h = int(size.get("height", 768))
    w = int(size.get("width", 768))
    mean = [float(v) for v in getattr(ip, "image_mean", (0.485, 0.456, 0.406))]
    std = [float(v) for v in getattr(ip, "image_std", (0.229, 0.224, 0.225))]
    return h, w, mean, std


def preprocess_differentiable(x, geom):
    """[1,3,H,W] in [0,1] -> the normalized [1,3,h,w] tensor Florence reads.

    A differentiable stand-in for the processor (bicubic resize to the fixed
    square, rescale, normalize). torch's antialiased bicubic and PIL's differ
    by rounding; preprocess_gap() measures the difference on a real image.
    Only the attacker's gradient path uses this: every scored inference goes
    through the real processor.
    """
    torch, F = rt.torch, rt.torch.nn.functional
    h, w, mean, std = geom
    if tuple(x.shape[-2:]) != (h, w):
        x = F.interpolate(x, size=(h, w), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0.0, 1.0)
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    sd = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / sd


def preprocess_gap(adapter, pil_img, geom):
    """max |processor(pil) - preprocess_differentiable(pil)| (normalized units)."""
    torch = rt.torch
    with torch.no_grad():
        ref = adapter.processor(text="<OD>", images=pil_img,
                                return_tensors="pt").pixel_values.float()
        mine = preprocess_differentiable(
            rt.pc.pil_to_tensor(pil_img, "cpu"), geom)
        if tuple(ref.shape) != tuple(mine.shape):
            raise RuntimeError(
                f"the processor returns pixel_values {tuple(ref.shape)} but "
                f"the differentiable preprocessing {tuple(mine.shape)}; the "
                f"attack would not reach the model the way inference does.")
        return float((ref - mine).abs().max())


def clean_self_labels(adapter, pil_img):
    """(input_ids, labels): the clean decode, used as the attack's labels."""
    a, torch = adapter.args, rt.torch
    inputs = adapter.processor(text="<OD>", images=pil_img,
                               return_tensors="pt")
    input_ids = inputs.input_ids.to(adapter.device)
    pixel_values = inputs.pixel_values.to(device=adapter.device,
                                          dtype=adapter.torch_dtype)
    with torch.no_grad():
        ids = adapter.model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=a.det_max_new_tokens, num_beams=a.num_beams,
            do_sample=False, repetition_penalty=a.repetition_penalty,
            length_penalty=a.length_penalty)
    return input_ids, ids[:1, :512].contiguous()


def _finite(t):
    return rt.torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def make_ce(adapter, geom, input_ids, labels):
    """ce(x) -> scalar tensor: teacher-forced CE of image x vs the labels."""
    def ce(x):
        pv = preprocess_differentiable(x, geom).to(adapter.torch_dtype)
        out = adapter.model(input_ids=input_ids, pixel_values=pv,
                            labels=labels)
        return _finite(out.loss.float())
    return ce


def oblivious_objective(ce):
    """J = -CE of the bare model: the attacker ignores the defense."""
    torch = rt.torch

    def objective(adv, need_grad):
        if not need_grad:
            with torch.no_grad():
                return -float(ce(adv)), None
        x = adv.detach().requires_grad_(True)
        loss = ce(x)
        (g,) = torch.autograd.grad(-loss, x)
        return -float(loss.detach()), _finite(g)

    return objective


def eot_objective(ce, branches, args):
    """J through the defense branches (BPDA), aggregated by --eot-loss.

    max : J = max_b(-CE_b) = -(CE of the least damaged branch); the gradient
          is that branch's alone. One no-grad forward per branch picks it,
          then one forward + backward.
    mean: J = -(mean of the branch CEs); one forward + backward per branch.
    Each branch forwards the deployed transform's exact value and
    differentiates the surrogate (ya.bpda_branch), as in p4_yolo.
    """
    torch = rt.torch
    use_max = args.eot_loss == "max" and len(branches) > 1

    def objective(adv, need_grad):
        adv = adv.detach()
        values = ya.deployed_values(adv, branches, args)
        if use_max or not need_grad:
            with torch.no_grad():
                losses = [float(ce(v)) for v in values]
            j = -min(losses) if use_max else -sum(losses) / len(losses)
            if not need_grad:
                return j, None
            picked = [min(range(len(losses)), key=losses.__getitem__)]
        else:
            picked, losses = list(range(len(branches))), []
        grads = []
        for i in picked:
            x = adv.clone().requires_grad_(True)
            loss = ce(ya.bpda_branch(x, branches[i], args, values[i]))
            (g,) = torch.autograd.grad(-loss, x)
            grads.append(_finite(g))
            if not use_max:
                losses.append(float(loss.detach()))
        if not use_max:
            j = -sum(losses) / len(losses)
        return j, torch.stack(grads).mean(dim=0)

    return objective


def native_frame(pil_img, device):
    """The attack variable: the native-resolution image in [0,1]."""
    from types import SimpleNamespace
    return SimpleNamespace(space="native",
                           x0=rt.pc.pil_to_tensor(pil_img, device),
                           to_model=lambda t: t,
                           to_pil=lambda u8: rt.Image.fromarray(u8))


def adaptive_signature(args, adapter, target, branches, files, image_ids):
    return rt.pc.config_signature(
        phase=4, experiment="adaptive_bpda", model=adapter.model_name,
        objective="self-labelled token CE, pixel space, native resolution",
        target=target, branches=sorted(branches), eps=args.eps,
        iters=args.pgd_iters, alpha=args.alpha, seed=args.seed,
        init=ya.PGD_INIT_SCHEME, surrogates=ya.SURROGATE_VERSION,
        restarts=args.restarts, eot_loss=args.eot_loss,
        median_surrogate=args.median_surrogate, quantize=args.quantize,
        attack_space=args.attack_space, pgd_select=args.pgd_select,
        jpeg_quality=args.jpeg_quality, median_kernel=args.median_kernel,
        gaussian_sigma=args.gaussian_sigma, tvm_weight=args.tvm_weight,
        tvm_iters=args.tvm_iters, nms_iou=args.nms_iou, n=len(files),
        ids=core.ids_digest(image_ids), device=str(adapter.device),
        **adapter.sig_fields(args))


def adaptive_bpda(adapter, args, coco_gt, files, image_ids):
    target = args.target
    ensembles, branches, paper = ya.resolve_target(target)
    is_ens = bool(ensembles)
    device = adapter.device
    infer = adapter.detection_fn("florence_det")
    ya.check_surrogates(device, args, branches)
    geom = processor_geometry(adapter.processor)
    OBL, ADP, META = ya.OBLIVIOUS_TAG, ya.ADAPTIVE_TAG, ya.META_KEY

    probe = rt.Image.open(os.path.join(args.image_dir, files[0])).convert("RGB")
    gap = preprocess_gap(adapter, probe, geom)
    print(f"[guard] differentiable preprocessing vs the processor on "
          f"{files[0]}: max |diff| = {gap:.4f} (normalized units; "
          f"1/255 is about 0.017)")

    conditions = (["clean"] + [f"clean+{b}" for b in branches]
                  + [OBL] + [f"{OBL}+{b}" for b in branches]
                  + [ADP] + [f"{ADP}+{b}" for b in branches])
    ckpt = core.make_checkpoint(args, "adaptive", adaptive_signature(
        args, adapter, target, branches, files, image_ids))

    rt.pc.print_banner(
        f"ADAPTIVE WHITE-BOX (Florence-2 <OD>): BPDA vs {paper} | "
        f"n={len(files)} | PGD eps={args.eps} x{args.pgd_iters} | "
        f"R={args.restarts} | eot={args.eot_loss} | pixel space", width=80)

    def defended(pil_img, tag, buckets, img_id):
        for b, img in core.build_branches(pil_img, device, args,
                                          branches).items():
            core.record_dets(buckets, f"{tag}+{b}", infer(img), img_id)

    def compute_one(pil_img, img_id):
        buckets = {}
        core.record_dets(buckets, "clean", infer(pil_img), img_id)
        defended(pil_img, "clean", buckets, img_id)
        input_ids, labels = clean_self_labels(adapter, pil_img)
        ce = make_ce(adapter, geom, input_ids, labels)
        frame = native_frame(pil_img, device)
        through = eot_objective(ce, branches, args)
        adv_obl, obl = ya.run_pgd(frame, oblivious_objective(ce), args, img_id)
        core.record_dets(buckets, OBL, infer(adv_obl), img_id)
        defended(adv_obl, OBL, buckets, img_id)
        adv_adp, adp = ya.run_pgd(frame, through, args, img_id)
        core.record_dets(buckets, ADP, infer(adv_adp), img_id)
        defended(adv_adp, ADP, buckets, img_id)
        buckets[META] = {
            "target_objective": {
                "clean": through(frame.x0, need_grad=False)[0],
                "oblivious": through(obl["adv_q"], need_grad=False)[0],
                "adaptive": adp["objective"]},
            "oblivious_bare_objective": obl["objective"],
            "label_tokens": int(labels.shape[1]),
            "restart_chosen": {"oblivious": obl["restart"],
                               "adaptive": adp["restart"]},
        }
        return buckets

    elapsed, computed, failures = core.run_per_image(
        args, "bpda", files, conditions, compute_one, ckpt,
        log_every=min(getattr(args, "log_every", 100), 25))
    ckpt.flush()
    _, cond = core.score_detection(args, ckpt, branches, ensembles, coco_gt,
                                   attack_tags=[OBL, ADP])

    head = {
        "clean_mAP": cond["clean"]["mAP"],
        "clean_defended_mAP": cond[f"clean+{target}"]["mAP"],
        "oblivious_attacked_undefended_mAP": cond[OBL]["mAP"],
        "oblivious_attacked_defended_mAP": cond[f"{OBL}+{target}"]["mAP"],
        "adaptive_attacked_undefended_mAP": cond[ADP]["mAP"],
        "adaptive_attacked_defended_mAP": cond[f"{ADP}+{target}"]["mAP"],
    }
    head["defense_recovery_oblivious"] = (
        head["oblivious_attacked_defended_mAP"]
        - head["oblivious_attacked_undefended_mAP"])
    head["defense_recovery_adaptive"] = (
        head["adaptive_attacked_defended_mAP"]
        - head["adaptive_attacked_undefended_mAP"])
    head["adaptive_minus_oblivious_defended"] = (
        head["adaptive_attacked_defended_mAP"]
        - head["oblivious_attacked_defended_mAP"])

    metas = [b[META] for b in ckpt.buckets.values() if META in b]
    tj = {k: [m["target_objective"][k] for m in metas]
          for k in ("clean", "oblivious", "adaptive")}
    below = [a < o for a, o in zip(tj["adaptive"], tj["oblivious"])]
    mean = lambda xs: sum(xs) / len(xs) if xs else None      # noqa: E731
    surrogate = {
        "definition": (f"J = -(teacher-forced cross-entropy against the "
                       f"clean decode), measured THROUGH {paper} "
                       f"(eot-loss {args.eot_loss}); lower = more damage. "
                       f"The attacker's own view, not a detection metric."),
        "images": len(metas), "mean_clean": mean(tj["clean"]),
        "mean_oblivious_adv": mean(tj["oblivious"]),
        "mean_adaptive_adv": mean(tj["adaptive"]),
        "frac_images_adaptive_below_oblivious":
            (sum(below) / len(below)) if below else None,
    }
    surrogates = {b: ya.surrogate_description(b, args) for b in branches}
    notes = [
        "Pixel space: the perturbation is added to the native-resolution "
        f"image in [0,1] and the uint8 output is clipped to the clean image "
        f"+/- {ya.uint8_budget(args.eps)}/255. Phase 3 attacked the "
        f"normalized 768x768 tensor (eps=0.03 there is about 1.7/255), so "
        f"these numbers are not comparable with the Phase-3 Florence runs.",
        f"The attacker's preprocessing is a differentiable stand-in for the "
        f"processor (antialiased bicubic resize, normalize); max difference "
        f"{gap:.4f} in normalized units on the first image. Scored inference "
        f"always uses the real processor.",
        "Objective: token-level cross-entropy against the model's own clean "
        "decode under teacher forcing; scored inference uses beam search, "
        "which the attack does not differentiate through.",
        "Detection scores are the geometric pseudo-scores of compute_score "
        "(Florence <OD> emits no confidences), as in every Florence table.",
        ("Ensemble objective 'max': each step raises the cross-entropy of "
         "the least damaged branch. A proxy for the NMS merge, which keeps "
         "a box if any branch keeps it." if is_ens and args.eot_loss == "max"
         else "Ensemble objective 'mean': the average branch cross-entropy."
         if is_ens else "Single transform: --eot-loss has no effect."),
        "BPDA: " + "; ".join(f"{b}: {d}" for b, d in surrogates.items())
        + ". Every forward is the deployed transform's exact value.",
        "Half precision on GPU, as in every Florence run; non-finite "
        "gradient entries are zeroed.",
    ]
    payload = {
        "phase": 4, "experiment": "adaptive_bpda",
        "model": adapter.model_name, "model_key": args.model_key,
        "smoke_test": bool(args.smoke_test), "target": target,
        "attack_flags": {k: getattr(args, k) for k in ya.ATTACK_FLAGS},
        "defense": {"name": paper, "key": target,
                    "kind": "ensemble" if is_ens else "solo",
                    "branches": branches, "nms_iou": args.nms_iou},
        "attack": {"type": "PGD", "eps": args.eps, "iters": args.pgd_iters,
                   "alpha": args.alpha, "seed": args.seed,
                   "restarts": args.restarts, "quantize": args.quantize,
                   "iterate_selection": args.pgd_select,
                   "init_scheme": ya.PGD_INIT_SCHEME,
                   "pixel_space": "raw [0,1], native resolution",
                   "objective": ("maximize the teacher-forced token "
                                 "cross-entropy against the clean decode")},
        "bpda": {"surrogates": surrogates, "eot_loss": args.eot_loss},
        "preprocess_gap_normalized": gap,
        "execution": {"deterministic": bool(getattr(args, "deterministic",
                                                    False)),
                      "dtype": str(adapter.torch_dtype),
                      "log_name": args.log_name},
        "config": {k: getattr(args, k) for k in
                   ("jpeg_quality", "median_kernel", "tvm_weight",
                    "tvm_iters", "gaussian_sigma", "nms_iou")},
        "model_config": adapter.sig_fields(args),
        "implementation_uncertainty": notes,
        "num_images": len(ckpt.buckets),
        "images_computed_this_run": computed,
        "select": args.select, "seed": args.seed,
        "image_ids": image_ids,
        "image_ids_sha1_16": core.ids_digest(image_ids),
        "device": str(device), "failures": failures, "seconds": elapsed,
        "seconds_per_image": elapsed / computed if computed else None,
        "headline": head, "surrogate_objective": surrogate,
        "conditions": cond,
    }
    md = [f"# Adaptive white-box: BPDA vs {paper} ({adapter.model_name}, "
          f"<OD>)\n"]
    md += core.header_lines(args, image_ids, device=device, extra=[
        f"- Target **{paper}** (`{target}`) = {' + '.join(branches)}",
        f"- PGD eps={args.eps} (pixel space), {args.pgd_iters} iters, "
        f"alpha={args.alpha:.5f}, {args.restarts} restart(s), seed "
        f"{args.seed}; ensemble objective `{args.eot_loss}`; iterate "
        f"selection `{args.pgd_select}`"])
    md.append("## Headline (COCO mAP@[.5:.95], geometric pseudo-scores)\n")
    md.append("| Condition | mAP |")
    md.append("|---|---|")
    for label, key in (
            ("clean, undefended", "clean_mAP"),
            (f"clean, {paper}", "clean_defended_mAP"),
            ("oblivious attack, undefended",
             "oblivious_attacked_undefended_mAP"),
            (f"oblivious attack, {paper}", "oblivious_attacked_defended_mAP"),
            ("adaptive attack, undefended", "adaptive_attacked_undefended_mAP"),
            (f"adaptive attack, {paper}", "adaptive_attacked_defended_mAP")):
        md.append(f"| {label} | {head[key]:.4f} |")
    md.append("")
    md.append(f"- Images where the adaptive J is below the oblivious J: "
              f"**{core.fmt(surrogate['frac_images_adaptive_below_oblivious'], 3)}**"
              f" of {surrogate['images']}\n")
    md.append("## All conditions\n")
    md.append("| Tag | mAP | AP50 |")
    md.append("|---|---|---|")
    for tag in sorted(cond):
        md.append(f"| `{tag}` | {cond[tag]['mAP']:.4f} | "
                  f"{cond[tag]['AP50']:.4f} |")
    md.append("")
    md.append("## Implementation notes\n")
    md += [f"- {n}" for n in notes]
    md.append("")
    md += core.timing_lines(elapsed, computed, len(files), str(device),
                            targets=(300, 1000, 5000))
    stem = ya.summary_stem(args, STEM_FLAG_DEFAULTS)
    core.write_per_image(args, stem, {
        "note": "per-image detection counts and the attacker's objectives",
        "counts": {str(i): {c: len(b.get(c, [])) for c in conditions}
                   for i, b in ckpt.buckets.items()},
        "attack_meta": {str(i): b.get(META)
                        for i, b in ckpt.buckets.items()}})
    core.write_summary(args, payload, stem)
    core.write_markdown(args, "\n".join(md), stem)
    print("\n" + "=" * 70)
    print(f"RESULT DIGEST: adaptive vs {paper} (full table in the .md)")
    print("=" * 70)
    for key, val in head.items():
        print(f"  {key:<38}: {val:+.4f}")
    print(f"  adaptive J below oblivious J on "
          f"{core.fmt(surrogate['frac_images_adaptive_below_oblivious'], 3)}"
          f" of images")
    core.print_runtime_estimate(elapsed, computed, str(device))
    return payload


# ======================================================================
# 3. Entry point.
# ======================================================================
def parse_args(argv=None):
    p = core.base_parser("Phase-4 Florence-2-base runs.", EXPERIMENTS)
    ya.add_attack_arguments(p, native_only=True)
    p.add_argument("--tracks", nargs="+", choices=list(ALL_TRACKS),
                   default=list(ALL_TRACKS),
                   help="Which Florence tracks to score (default: both, from "
                        "one model load).")
    g = p.add_argument_group("generation (Phase-3 values)")
    g.add_argument("--det-max-new-tokens", type=int, default=512)
    g.add_argument("--ocr-max-new-tokens", type=int, default=256)
    g.add_argument("--num-beams", type=int, default=5)
    g.add_argument("--repetition-penalty", type=float, default=1.8)
    g.add_argument("--length-penalty", type=float, default=1.0)
    g.add_argument("--full-gen", action="store_true",
                   help="With --smoke-test, keep the real generation config "
                        "(5 beams) instead of the fast one.")
    m = p.add_argument_group("model (Phase-3 values)")
    m.add_argument("--florence-model", default="microsoft/Florence-2-base")
    m.add_argument("--florence-revision", default="refs/pr/26")

    args = p.parse_args(argv)
    core.finalize_args(args, "florence")
    ya.finalize_attack_args(args)
    if args.experiment == "adaptive_bpda":
        args.tracks = ["florence_det"]        # the attack is on <OD> only
    if args.smoke_test and not args.full_gen:
        # 5 beams x 512 tokens on CPU is minutes per image. The smoke test only
        # proves the plumbing, and its output is stamped smoke_test=true, so a
        # cheaper decode is safe. --full-gen keeps the exact Phase-3 decode.
        args.num_beams = 1
        args.det_max_new_tokens = 128
        args.ocr_max_new_tokens = 64
    return args


def hf_hub_cache():
    """(hub cache dir, how it was found), the dir transformers will use.

    huggingface_hub.constants.HF_HUB_CACHE when importable (it honours
    HF_HUB_CACHE and HF_HOME, e.g. ./weights/hf from scripts/env.sh); else
    the same rules by hand: $HF_HUB_CACHE, $HF_HOME/hub, then
    ~/.cache/huggingface/hub.
    """
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE, "huggingface_hub.constants.HF_HUB_CACHE"
    except Exception:  # noqa: BLE001 (not installed, or a broken install)
        pass
    if os.environ.get("HF_HUB_CACHE"):
        return os.path.expanduser(os.environ["HF_HUB_CACHE"]), "$HF_HUB_CACHE"
    if os.environ.get("HF_HOME"):
        return (os.path.join(os.path.expanduser(os.environ["HF_HOME"]), "hub"),
                "$HF_HOME/hub")
    return os.path.expanduser("~/.cache/huggingface/hub"), "default"


def hf_offline():
    """True when HF_HUB_OFFLINE (or the older TRANSFORMERS_OFFLINE) is on."""
    return core.env_flag("HF_HUB_OFFLINE") or core.env_flag(
        "TRANSFORMERS_OFFLINE")


def hf_cached(hub, model, revision):
    """(cached at `revision`?, repo dir, detail) from the hub cache layout.

    models--<org>--<name>/refs/<revision> holds the commit hash and
    snapshots/<hash>/ the files; an offline load needs both. A 40-hex
    revision is a commit hash and is looked up in snapshots/ directly.
    """
    repo = os.path.join(hub, "models--" + model.replace("/", "--"))
    if not os.path.isdir(repo):
        return False, repo, "not in the cache"
    commit = revision if (len(revision) == 40 and all(
        c in "0123456789abcdef" for c in revision)) else None
    if commit is None:
        ref = os.path.join(repo, "refs", *revision.split("/"))
        try:
            with open(ref) as f:
                commit = f.read().strip()
        except OSError:
            return False, repo, f"cached, but not at revision {revision}"
    snap = os.path.join(repo, "snapshots", commit)
    if not os.path.isfile(os.path.join(snap, "config.json")):
        return False, repo, (f"revision {revision} -> {commit[:12]} has no "
                             f"complete snapshot")
    return True, repo, f"revision {revision} -> {commit[:12]}"


def preflight_weights(args):
    """[(label, present, path checked, offline reason)] for core.preflight."""
    hub, how = hf_hub_cache()
    present, repo, detail = hf_cached(hub, args.florence_model,
                                      args.florence_revision)
    offline = (f"HF_HUB_OFFLINE is set and {detail}: fill the cache on a "
               f"machine with internet (same HF_HOME), or unset "
               f"HF_HUB_OFFLINE" if hf_offline() else None)
    return [(f"{args.florence_model}@{args.florence_revision}", present,
             f"{repo} ({how}; {detail})", offline)]


def main(argv=None):
    args = parse_args(argv)
    if args.preflight:
        needs_coco = "florence_det" in args.tracks
        return core.preflight(
            args,
            modules=[("numpy", "all"), ("PIL", "all"), ("torch", "all"),
                     ("transformers", "Florence-2"),
                     ("pycocotools", "COCO mAP"
                      if needs_coco else "not needed for the OCR track")],
            weights=preflight_weights(args))

    device = core.load_runtime(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with core.run_log(core.log_path(args), argv=sys.argv,
                      extra={"experiment": args.experiment, "n": args.n,
                             "tracks": ",".join(args.tracks),
                             "device": str(device),
                             "visible": os.environ.get(
                                 "CUDA_VISIBLE_DEVICES"),
                             "smoke_test": args.smoke_test}):
        args.image_dir = core.resolve_image_dir(args.image_dir)
        args.ann_file = core.resolve_ann_file(args.ann_file)
        files, image_ids = core.select_images(args.image_dir, args.n, args.seed,
                                             args.select,
                                             allow_short=args.smoke_test)
        args.n = len(files)

        rt.pc.print_banner(f"PHASE 4 — FLORENCE-2 — {args.experiment}", width=80)
        if args.smoke_test:
            print(f"*** SMOKE TEST: {args.n} images on {device}. The numbers "
                  f"below are NOT results. ***")
        print(f"  run id    : {args.run_id}")
        print(f"  images    : {args.n} from {args.image_dir} "
              f"(--select {args.select}, seed {args.seed})")
        print(f"  id digest : {core.ids_digest(image_ids)}")
        print(f"  tracks    : {args.tracks}")
        print(f"  device    : {device}")
        print(f"  output    : {args.output_dir}")
        print(f"  log       : {args.log_name}")

        coco_gt = None
        if "florence_det" in args.tracks:
            from pycocotools.coco import COCO
            coco_gt = COCO(args.ann_file)

        adapter = FlorenceAdapter(args, device, args.tracks)
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
