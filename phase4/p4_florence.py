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
# NOT IN THIS FILE (yet): the FGSM/PGD/Patch attacks. They live in
# archive/Phase3/{FGSM,PGD,Patch}_Phase3_Florence_v2.py and
# archive/Phase3/run_survey_florence_{detection,ocr}.py. Port them in when a
# Phase-4 attack experiment is actually defined — copying them here now would
# add several hundred lines of untested code with nothing calling it.
# ======================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p4_core as core                                     # noqa: E402
from p4_core import rt                                     # noqa: E402

EXPERIMENTS = ("clean_control",)
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
# 2. Entry point.
# ======================================================================
def parse_args(argv=None):
    p = core.base_parser("Phase-4 Florence-2-base runs.", EXPERIMENTS)
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
    if args.smoke_test and not args.full_gen:
        # 5 beams x 512 tokens on CPU is minutes per image. The smoke test only
        # proves the plumbing, and its output is stamped smoke_test=true, so a
        # cheaper decode is safe. --full-gen keeps the exact Phase-3 decode.
        args.num_beams = 1
        args.det_max_new_tokens = 128
        args.ocr_max_new_tokens = 64
    return args


def main(argv=None):
    args = parse_args(argv)
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    cached = os.path.isdir(os.path.join(
        hub, "models--" + args.florence_model.replace("/", "--")))
    if args.preflight:
        needs_coco = "florence_det" in args.tracks
        return core.preflight(
            args,
            modules=[("numpy", "all"), ("PIL", "all"), ("torch", "all"),
                     ("transformers", "Florence-2"),
                     ("pycocotools", "COCO mAP"
                      if needs_coco else "not needed for the OCR track")],
            weights=[(args.florence_model, cached)])

    device = core.load_runtime(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with core.run_log(os.path.join(args.output_dir, "run.log"), argv=sys.argv,
                      extra={"experiment": args.experiment, "n": args.n,
                             "tracks": ",".join(args.tracks),
                             "device": str(device),
                             "smoke_test": args.smoke_test}):
        args.image_dir = core.resolve_image_dir(args.image_dir)
        args.ann_file = core.resolve_ann_file(args.ann_file)
        files, image_ids = core.select_images(args.image_dir, args.n, args.seed,
                                             args.select)
        args.n = len(files)

        rt.pc.print_banner(f"PHASE 4 — FLORENCE-2 — {args.experiment}", width=80)
        if args.smoke_test:
            print("*** SMOKE TEST: 2 images on CPU, reduced decode. The numbers "
                  "below are NOT results. ***")
        print(f"  run id    : {args.run_id}")
        print(f"  images    : {args.n} from {args.image_dir} "
              f"(--select {args.select}, seed {args.seed})")
        print(f"  id digest : {core.ids_digest(image_ids)}")
        print(f"  tracks    : {args.tracks}")
        print(f"  device    : {device}")
        print(f"  output    : {args.output_dir}")

        coco_gt = None
        if "florence_det" in args.tracks:
            from pycocotools.coco import COCO
            coco_gt = COCO(args.ann_file)

        adapter = FlorenceAdapter(args, device, args.tracks)
        adapter.load(coco_gt)

        if args.experiment == "clean_control":
            core.clean_control(adapter, args, coco_gt, files, image_ids)
        else:  # pragma: no cover — argparse restricts the choices
            raise ValueError(f"unknown experiment {args.experiment}")

        core.finish(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
