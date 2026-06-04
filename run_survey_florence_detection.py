#!/usr/bin/env python3
"""
run_survey_florence_detection.py — Florence-2 object-detection robustness survey.
(For Lokendra.)

One process, one model load, all requested attacks run SEQUENTIALLY. Surveys
the Tier-1 / Tier-2 defense bank (via phase3_common.apply_survey_defenses) and,
optionally, a novel SIGN-approximation defense.

Adapted from the frozen *_v2 Florence notebooks
(FGSM_/PGD_/Patch_Phase3_Florence_v2.py): run_inference, FLORENCE_TO_COCO,
_map_label, _compute_score, non_max_suppression, the three attacks, and the
per-image buckets + hard-assert pattern are reused VERBATIM. Only the driver
(argparse, sequential attack loop, survey defense bank, config-keyed
checkpoint, minimal logging, novel defense) is new.

====================================================================
CITATIONS
====================================================================
Attacks
  FGSM   — Goodfellow, Shlens & Szegedy, ICLR 2015.
  PGD    — Madry et al., ICLR 2018.
  Patch  — Brown et al., 2017 (Adversarial Patch).

Defenses
  jpeg        — Dziugaite et al., 2016 (JPEG compression as a defense).
  median      — Xu et al., NDSS 2018 (feature squeezing).
  bit_depth   — Xu et al., NDSS 2018 (feature squeezing).
  gaussian    — Xu et al., NDSS 2018 (feature squeezing / spatial smoothing).
  tvm         — Guo et al., ICLR 2018 (total-variation minimization).
  nlm         — Buades et al., CVPR 2005 (non-local means).
  svd         — spectral (low-rank SVD) filter.
  bilateral   — Tomasi & Manduchi, ICCV 1998.
  dithering   — ordered (Bayer) dithering; a fast DETERMINISTIC stand-in for
                serial Floyd-Steinberg error diffusion.
  anisotropic — Perona & Malik, IEEE PAMI 1990 (edge-preserving diffusion).
  bm3d        — Dabov et al., IEEE TIP 2007 (OPTIONAL; dropped if not installed).

Novel (--novel)
  sign_approx — APPROXIMATION OF SIGN (arXiv:2605.27927, 2026, preprint).
                This is NOT the full method: it approximates the salient-pixel
                sign-suppression idea by detecting the top-0.5% highest
                Sobel-gradient pixels and replacing each with its local 3x3
                median. Pure numpy/torch; no extra dependencies. The output row
                is labeled 'sign_approx' to flag that it is an approximation.
  (PAD — CVPR 2024, arXiv:2404.16452, SAM-based patch defense — and
   XAIAD-YOLO — Future Generation Computer Systems 2026,
   DOI 10.1016/j.future.2025.108356 — are cited for completeness; only
   sign_approx is implemented here.)
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import warnings

# ======================================================================
# 1. GPU isolation — MUST happen BEFORE importing torch / phase3_common.
#    Scan sys.argv for --gpu manually (argparse runs after torch import).
# ======================================================================
def _early_gpu_isolation() -> None:
    gpu = None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--gpu" and i + 1 < len(argv):
            try:
                gpu = int(argv[i + 1])
            except ValueError:
                gpu = None
            break
        if a.startswith("--gpu="):
            try:
                gpu = int(a.split("=", 1)[1])
            except ValueError:
                gpu = None
            break

    if gpu is not None and gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (requested)")
        return

    # gpu is None or < 0 -> pick the freest GPU via inline nvidia-smi
    # (same logic as the *_v2 Florence files).
    import subprocess
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpu_free = []
        for line in smi.stdout.strip().split("\n"):
            idx, free = line.split(",")
            gpu_free.append((int(idx.strip()), int(free.strip())))
        gpu_free.sort(key=lambda x: x[1], reverse=True)
        selected = str(gpu_free[0][0]) if gpu_free else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (auto-freest)")
    except Exception as e:  # noqa: BLE001
        print(f"[GPU] nvidia-smi failed ({e}); using default CUDA_VISIBLE_DEVICES")


_early_gpu_isolation()

import torch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, AutoModelForCausalLM  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

import phase3_common as pc  # noqa: E402

warnings.filterwarnings("ignore")


# ======================================================================
# 2. Argument parsing (flags mirror the YOLO survey script).
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Florence-2 object-detection adversarial-robustness survey."
    )
    p.add_argument("--gpu", type=int, default=-1,
                   help="GPU index; <0 picks the freest GPU (handled pre-import).")
    p.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "patch"],
                   choices=["fgsm", "pgd", "patch"],
                   help="Attacks to run SEQUENTIALLY in this one process.")
    p.add_argument("--tier", choices=["tier1", "survey"], default="survey",
                   help="Defense set: tier1 = locked TIER1 set; survey = full SURVEY set.")
    p.add_argument("--num-images", type=int, default=1000,
                   help="Number of images (sorted) to evaluate.")
    p.add_argument("--image-dir", default="./Dataset/val2017",
                   help="COCO val image directory.")
    p.add_argument("--ann-file", default="./Dataset/annotations/instances_val2017.json",
                   help="COCO instances annotation file.")
    p.add_argument("--eps", type=float, default=0.03,
                   help="Epsilon for FGSM/PGD (normalized-pixel space).")
    p.add_argument("--pgd-iters", type=int, default=10, help="PGD iterations.")
    p.add_argument("--checkpoint-every", type=int, default=100,
                   help="Flush the config-keyed checkpoint every N images.")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable on-disk checkpointing (pure in-memory run).")
    p.add_argument("--novel", action="store_true",
                   help="Add the novel SIGN-approximation defense row ('sign_approx').")
    p.add_argument("--nms-iou", type=float, default=0.5,
                   help="IoU threshold for inference NMS and ensemble merging.")
    return p.parse_args()


ARGS = parse_args()

# Resolved config constants
IMAGE_DIR = ARGS.image_dir
ANN_FILE = ARGS.ann_file
NUM_IMAGES = ARGS.num_images
EPSILON = ARGS.eps
PGD_ITERS = ARGS.pgd_iters
PGD_ALPHA = EPSILON / 4.0
PATCH_SIZE = pc.PATCH_SIZE
PATCH_OPT_ITERS = pc.PATCH_OPT_ITERS
PATCH_LR = pc.PATCH_LR

# Defense hyperparams (match the *_v2 Florence config exactly)
JPEG_QUALITY = 75
MEDIAN_KERNEL = 3
TVM_WEIGHT = 0.05
TVM_ITERS = 200
GAUSSIAN_SIGMA = 1.0
NMS_IOU_THRESHOLD = ARGS.nms_iou
ENSEMBLE_NMS_IOU = ARGS.nms_iou

OUTPUT_DIR = "/home/king/Documents/Projects/Adversarial_Attack_Defence_Paper/results_survey_florence_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map attack key -> COCO condition tag
ATTACK_TAGS = {
    "fgsm": f"fgsm_eps{EPSILON}",
    "pgd": f"pgd_eps{EPSILON}",
    "patch": "patch",
}

# ======================================================================
# 3. Device + model (loaded ONCE; reused by every attack).
# ======================================================================
assert torch.cuda.is_available(), "CUDA required for the Florence survey."
device = torch.device("cuda:0")
torch_dtype = torch.float16

model_name = "microsoft/Florence-2-base"
revision = "refs/pr/26"

pc.print_banner("Florence-2 Detection Robustness Survey", width=70)
print(f"Attacks: {ARGS.attacks} | tier: {ARGS.tier} | images: {NUM_IMAGES} | novel: {ARGS.novel}")
print(f"Device: {device} | dtype: {torch_dtype} | output: {OUTPUT_DIR}")

processor = AutoProcessor.from_pretrained(model_name, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, revision=revision,
    torch_dtype=torch_dtype, trust_remote_code=True,
).to(device).eval()

IMG_MEAN = torch.tensor(processor.image_processor.image_mean,
                        device=device, dtype=torch_dtype).view(1, 3, 1, 1)
IMG_STD = torch.tensor(processor.image_processor.image_std,
                       device=device, dtype=torch_dtype).view(1, 3, 1, 1)

coco_gt = COCO(ANN_FILE)
categories = coco_gt.loadCats(coco_gt.getCatIds())
category_mapping = {c["name"]: c["id"] for c in categories}

files = sorted(os.listdir(IMAGE_DIR))
if NUM_IMAGES is not None:
    files = files[:NUM_IMAGES]
num_images = len(files)
processed_ids_all = sorted({int(os.path.splitext(f)[0]) for f in files})
print(f"COCO: {len(category_mapping)} categories | {num_images} images selected.")


# ======================================================================
# 4. Florence-2 -> COCO inference pipeline (VERBATIM from the *_v2 files).
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
    "houseplant": "potted plant", "plant": "potted plant", "flower pot": "potted plant",
    "flowerpot": "potted plant",
    "wine bottle": "bottle", "beer bottle": "bottle", "water bottle": "bottle",
    "drinking glass": "wine glass", "glass": "wine glass", "goblet": "wine glass",
    "pocketknife": "knife", "kitchen knife": "knife", "butter knife": "knife",
    "hair dryer": "hair drier", "hairdryer": "hair drier", "blow dryer": "hair drier",
    "wristwatch": "clock", "wall clock": "clock", "alarm clock": "clock",
    "bag": "handbag", "purse": "handbag", "wristlet": "handbag",
    "briefcase": "suitcase", "luggage": "suitcase", "travel bag": "suitcase",
    "backpack bag": "backpack",
    "traffic signal": "traffic light",
    "fire plug": "fire hydrant",
}


def _map_label(label):
    if label in category_mapping: return label
    m = FLORENCE_TO_COCO.get(label)
    if m and m in category_mapping: return m
    lo = label.lower()
    if lo in category_mapping: return lo
    m2 = FLORENCE_TO_COCO.get(lo)
    if m2 and m2 in category_mapping: return m2
    return None


def _compute_score(box, img_w, img_h):
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
    if not boxes: return [], [], []
    boxes = np.array(boxes)
    idxs = np.argsort(scores)[::-1]
    keep, kl, ks = [], [], []
    for i in idxs:
        sup = False
        for j in keep:
            if labels[i] == labels[j] and _box_iou_local(boxes[i], boxes[j]) > iou_thr:
                sup = True; break
        if not sup:
            keep.append(i); kl.append(labels[i]); ks.append(scores[i])
    return boxes[keep].tolist(), kl, ks


def run_inference(pil_img):
    img_w, img_h = pil_img.size
    with torch.no_grad():
        inputs = processor(text="<OD>", images=pil_img, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype)
        gen_ids = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=512, num_beams=5, do_sample=False,
            repetition_penalty=1.8, length_penalty=1.0,
        )
        txt = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            txt, task="<OD>", image_size=(img_w, img_h)
        ) or {}
    od = parsed.get("<OD>", {})
    bboxes, labels = od.get("bboxes", []), od.get("labels", [])
    scores = [_compute_score(b, img_w, img_h) for b in bboxes]
    kb, kl, ks = non_max_suppression(bboxes, labels, scores, iou_thr=NMS_IOU_THRESHOLD)
    results = []
    for box, lab, sc in zip(kb, kl, ks):
        mp = _map_label(lab)
        if mp is None: continue
        cid = category_mapping[mp]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: continue
        results.append({"bbox": [x1, y1, w, h], "category_id": cid, "score": sc})
    return results


# ======================================================================
# 5. Attacks (VERBATIM from the *_v2 Florence files; normalized-pixel space).
# ======================================================================
def fgsm_attack(pil_img, eps=EPSILON):
    orig_size = pil_img.size
    inputs = processor(text="<OD>", images=pil_img, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype)

    with torch.no_grad():
        target_ids = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=512, num_beams=5, do_sample=False,
            repetition_penalty=1.8, length_penalty=1.0,
        )
    target_ids = target_ids[:1, :].contiguous()
    if target_ids.size(1) > 512:
        target_ids = target_ids[:, :512].contiguous()

    pv_adv = pixel_values.clone().detach().requires_grad_(True)
    out = model(input_ids=input_ids, pixel_values=pv_adv, labels=target_ids)
    out.loss.backward()

    grad_sign = pv_adv.grad.sign()
    adv = pixel_values.detach() + eps * grad_sign
    adv = torch.clamp(adv, -2.5, 2.5)

    adv_denorm = (adv.squeeze(0) * IMG_STD.squeeze(0) + IMG_MEAN.squeeze(0)).clamp(0.0, 1.0)
    adv_np = (adv_denorm.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    if adv_pil.size != orig_size:
        adv_pil = adv_pil.resize(orig_size, Image.BICUBIC)
    return adv_pil


def pgd_attack(pil_img, eps=EPSILON, iters=PGD_ITERS, alpha=PGD_ALPHA):
    orig_size = pil_img.size
    inputs = processor(text="<OD>", images=pil_img, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype)

    with torch.no_grad():
        target_ids = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=512, num_beams=5, do_sample=False,
            repetition_penalty=1.8, length_penalty=1.0,
        )
    target_ids = target_ids[:1, :].contiguous()
    if target_ids.size(1) > 512:
        target_ids = target_ids[:, :512].contiguous()

    delta = torch.empty_like(pixel_values).uniform_(-eps, eps)
    adv = torch.clamp(pixel_values + delta, -2.5, 2.5).detach()

    for _ in range(iters):
        adv = adv.detach().requires_grad_(True)
        out = model(input_ids=input_ids, pixel_values=adv, labels=target_ids)
        if adv.grad is not None:
            adv.grad.zero_()
        out.loss.backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            d = torch.clamp(adv - pixel_values, min=-eps, max=eps)
            adv = torch.clamp(pixel_values + d, -2.5, 2.5).detach()

    adv_denorm = (adv.squeeze(0) * IMG_STD.squeeze(0) + IMG_MEAN.squeeze(0)).clamp(0.0, 1.0)
    adv_np = (adv_denorm.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    if adv_pil.size != orig_size:
        adv_pil = adv_pil.resize(orig_size, Image.BICUBIC)
    return adv_pil


def patch_attack(pil_img, patch_size=PATCH_SIZE,
                 iters=PATCH_OPT_ITERS, lr=PATCH_LR):
    orig_size = pil_img.size
    inputs = processor(text="<OD>", images=pil_img, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype)

    with torch.no_grad():
        target_ids = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=512, num_beams=5, do_sample=False,
            repetition_penalty=1.8, length_penalty=1.0,
        )
    target_ids = target_ids[:1, :].contiguous()
    if target_ids.size(1) > 512:
        target_ids = target_ids[:, :512].contiguous()

    H, W = pixel_values.shape[-2:]
    top, left = pc.center_patch_coords(H, W, patch_size)

    torch.manual_seed(0)
    patch01 = (torch.rand(3, patch_size, patch_size,
                          device=device, dtype=torch_dtype) * 0.5 + 0.25)
    patch01 = patch01.detach().clone()
    patch01.requires_grad_(True)
    optim = torch.optim.Adam([patch01], lr=lr)

    mean3 = IMG_MEAN.view(3, 1, 1)
    std3 = IMG_STD.view(3, 1, 1)

    for _ in range(iters):
        optim.zero_grad()
        patch_norm = (patch01.clamp(0.0, 1.0) - mean3) / std3
        composed = pixel_values.clone()
        composed[:, :, top:top + patch_size, left:left + patch_size] = patch_norm.unsqueeze(0)
        out = model(input_ids=input_ids, pixel_values=composed, labels=target_ids)
        (-out.loss).backward()
        optim.step()

    with torch.no_grad():
        patch_final = patch01.clamp(0.0, 1.0)
        patch_norm_final = (patch_final - mean3) / std3
        adv = pixel_values.clone()
        adv[:, :, top:top + patch_size, left:left + patch_size] = patch_norm_final.unsqueeze(0)

    adv_denorm = (adv.squeeze(0) * IMG_STD.squeeze(0) + IMG_MEAN.squeeze(0)).clamp(0.0, 1.0)
    adv_np = (adv_denorm.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    if adv_pil.size != orig_size:
        adv_pil = adv_pil.resize(orig_size, Image.BICUBIC)
    return adv_pil


ATTACK_FNS = {"fgsm": fgsm_attack, "pgd": pgd_attack, "patch": patch_attack}


# ======================================================================
# 6. Novel defense — SIGN APPROXIMATION (arXiv:2605.27927).
#    APPROXIMATION ONLY: detect the top-0.5% highest Sobel-gradient pixels
#    and replace each with its local 3x3 median. Pure numpy; no extra deps.
#    Output row is labeled 'sign_approx' to flag the approximation.
# ======================================================================
def sign_approx(pil_img, frac=0.005):
    arr = np.array(pil_img.convert("RGB")).astype(np.float32)
    H, W, _ = arr.shape
    gray = arr.mean(axis=2)

    # Sobel gradient magnitude (reflect-padded so shapes stay [H,W]).
    g = np.pad(gray, 1, mode="reflect")
    gx = (g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]
          - g[:-2, :-2] - 2 * g[1:-1, :-2] - g[2:, :-2])
    gy = (g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]
          - g[:-2, :-2] - 2 * g[:-2, 1:-1] - g[:-2, 2:])
    mag = np.sqrt(gx * gx + gy * gy)

    # Top-`frac` highest-gradient pixels.
    k = max(1, int(round(frac * H * W)))
    thresh = np.partition(mag.ravel(), -k)[-k]
    mask = mag >= thresh  # [H,W] bool

    # Local 3x3 median per channel (reflect-padded), vectorized via stacking
    # the 9 shifts of the padded array.
    out = arr.copy()
    padded = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    neigh = np.stack(
        [padded[dy:dy + H, dx:dx + W, :]
         for dy in range(3) for dx in range(3)],
        axis=0,
    )                                    # [9,H,W,C]
    med = np.median(neigh, axis=0)       # [H,W,C]
    out[mask] = med[mask]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ======================================================================
# 7. Survey defense bank.
#   - survey -> full SURVEY set via apply_survey_defenses.
#   - tier1  -> subset to pc.TIER1_SOLOS (and the survey producer's keys).
#   Always derive defense_names from the producer's ACTUAL returned keys.
#   With --novel, append 'sign_approx' (built per-image inside the loop).
# ======================================================================
def build_defenses(pil_img, img_id):
    """Return {defense_name: PIL} for the requested tier, plus sign_approx.

    For tier1 we still call apply_survey_defenses (so 'svd'/'bilateral' etc.
    are produced consistently) and then subset to TIER1_SOLOS keys that were
    actually returned. The novel sign_approx row is added (with try/except) so
    that a failure of the novel defense never kills the image.
    """
    survey_defs = pc.apply_survey_defenses(
        pil_img, device,
        jpeg_quality=JPEG_QUALITY, median_kernel=MEDIAN_KERNEL,
        tvm_weight=TVM_WEIGHT, tvm_iters=TVM_ITERS,
        gaussian_sigma=GAUSSIAN_SIGMA, seed=img_id,
    )
    if ARGS.tier == "tier1":
        defs = {k: v for k, v in survey_defs.items() if k in pc.TIER1_SOLOS}
    else:
        defs = dict(survey_defs)

    if ARGS.novel:
        try:
            defs["sign_approx"] = sign_approx(pil_img)
        except Exception as exc:  # noqa: BLE001
            global _SIGN_WARNED
            if not _SIGN_WARNED:
                print(f"[novel] sign_approx failed ({exc}); skipping the row.")
                _SIGN_WARNED = True
    return defs


_SIGN_WARNED = False
_NOVEL_SOLOS = ["sign_approx"] if ARGS.novel else []


def select_registry():
    """Return (base_solos, base_ensembles) for the chosen tier."""
    if ARGS.tier == "tier1":
        return list(pc.TIER1_SOLOS), dict(pc.TIER1_ENSEMBLES)
    return list(pc.SURVEY_SOLOS), dict(pc.SURVEY_ENSEMBLES)


# ======================================================================
# 8. Per-attack runner.
# ======================================================================
def ranked_table(eval_stats, attack_tag, defense_names, ensembles):
    """Print the ranked defense table (same format as the *_v2 base script)."""
    clean_ap = float(eval_stats["clean"][0])
    atk_ap = float(eval_stats[attack_tag][0])
    attack_drop = clean_ap - atk_ap

    all_defense_names = list(defense_names) + list(ensembles.keys())
    rows = []
    for dn in all_defense_names:
        tag = f"{attack_tag}+{dn}"
        if tag not in eval_stats:
            continue
        def_ap = float(eval_stats[tag][0])
        rec = def_ap - atk_ap
        rec_pct = (100.0 * rec / attack_drop) if attack_drop > 0 else 0.0
        kind = "ENSEMBLE" if dn in ensembles else "solo"
        rows.append((dn, kind, def_ap, rec, rec_pct))
    rows.sort(key=lambda r: r[3], reverse=True)

    pc.print_banner(f"ATTACKED vs RECOVERED — {attack_tag} (FLORENCE-2)", width=100)
    print(f"  Clean baseline   : {clean_ap:.4f}")
    print(f"  Attacked         : {atk_ap:.4f}")
    print(f"  Attack damage    : {attack_drop:+.4f}\n")
    print("-" * 100)
    print(f"  {'Rank':<5} {'Defense':<28} {'Kind':<10} {'Attacked':>9} "
          f"{'Recovered':>10} {'Δ mAP':>9} {'Rec%':>7} {'Verdict':>10}")
    print("-" * 100)
    for rank, (name, kind, def_ap, rec, rec_pct) in enumerate(rows, 1):
        if rec > 0.005:
            verdict = "RECOVERS"
        elif rec > 0:
            verdict = "marginal"
        elif rec > -0.005:
            verdict = "neutral"
        else:
            verdict = "HURTS"
        print(f"  {rank:<5} {name:<28} {kind:<10} {atk_ap:>9.4f} {def_ap:>10.4f} "
              f"{rec:>+9.4f} {rec_pct:>+6.1f}% {verdict:>10}")
    print("-" * 100)
    return rows, clean_ap, atk_ap, attack_drop


def run_attack(attack_key):
    attack_tag = ATTACK_TAGS[attack_key]
    attack_fn = ATTACK_FNS[attack_key]

    pc.print_banner(f"Survey attack: {attack_key} ({attack_tag})", width=70)

    # --- Discover the actual defense_names from a real producer call -------
    # (Build on the first image so bm3d-dropping etc. is reflected, then add
    #  the novel row name if requested.)
    probe_img = Image.open(os.path.join(IMAGE_DIR, files[0])).convert("RGB")
    probe_id = int(os.path.splitext(files[0])[0])
    probe_defs = build_defenses(probe_img, probe_id)
    defense_names = list(probe_defs.keys())

    # Filter ensembles to those whose EVERY member is in defense_names.
    _, base_ensembles = select_registry()
    ensembles = {
        name: members for name, members in base_ensembles.items()
        if all(m in defense_names for m in members)
    }
    print(f"[{attack_key}] defenses ({len(defense_names)}): {defense_names}")
    print(f"[{attack_key}] ensembles ({len(ensembles)}): {list(ensembles.keys())}")

    # --- SAFE config-keyed checkpoint --------------------------------------
    sig = pc.config_signature(
        model=model_name, attack=attack_tag,
        defenses=sorted(defense_names), eps=EPSILON,
        pgd_iters=PGD_ITERS, pgd_alpha=PGD_ALPHA,
        patch_size=PATCH_SIZE, patch_iters=PATCH_OPT_ITERS, patch_lr=PATCH_LR,
        num_images=num_images, image_dir=IMAGE_DIR,
    )
    ckpt = pc.SurveyCheckpoint(
        os.path.join(OUTPUT_DIR, f"checkpoint_{attack_key}.pkl"),
        sig, flush_every=ARGS.checkpoint_every, enabled=not ARGS.no_checkpoint,
    )

    expected_tags = (
        ["clean"]
        + [f"clean+{d}" for d in defense_names]
        + [attack_tag]
        + [f"{attack_tag}+{d}" for d in defense_names]
    )

    # --- Main loop ---------------------------------------------------------
    start = time.time()
    atk_running, best_running = [], []  # for the 100-image heartbeat
    processed = 0
    failures = []

    for i, fname in enumerate(files):
        img_id = int(os.path.splitext(fname)[0])
        if ckpt.has(img_id):
            processed += 1
            # Recompute heartbeat stats lazily only when printing; cached
            # images contribute their stored values below if needed.
            if (i + 1) % 100 == 0:
                elapsed = (time.time() - start) / 60.0
                atk_avg = float(np.mean(atk_running)) if atk_running else float("nan")
                best = float(np.max(best_running)) if best_running else float("nan")
                print(f"[{attack_tag}] {i + 1}/{num_images} | "
                      f"attacked={atk_avg:.4f} | best={best:.4f} | {elapsed:.1f}m")
            continue

        try:
            pil_img = Image.open(os.path.join(IMAGE_DIR, fname)).convert("RGB")
            buckets = {}

            def _record(tag, dets):
                for d in dets:
                    d["image_id"] = img_id
                buckets[tag] = dets

            # 1. Clean baseline
            _record("clean", run_inference(pil_img))
            # 2. Clean + each survey defense
            clean_defs = build_defenses(pil_img, img_id)
            for dn in defense_names:
                if dn in clean_defs:
                    _record(f"clean+{dn}", run_inference(clean_defs[dn]))
            # 3. Attacked
            adv = attack_fn(pil_img)
            _record(attack_tag, run_inference(adv))
            # 4. Attacked + each survey defense
            atk_defs = build_defenses(adv, img_id)
            for dn in defense_names:
                if dn in atk_defs:
                    _record(f"{attack_tag}+{dn}", run_inference(atk_defs[dn]))

            ckpt.put(img_id, buckets)
            processed += 1

            # heartbeat stats (count of attacked dets vs best defended count)
            atk_running.append(len(buckets.get(attack_tag, [])))
            best_running.append(max(
                (len(buckets.get(f"{attack_tag}+{dn}", [])) for dn in defense_names),
                default=0,
            ))
        except Exception as exc:  # noqa: BLE001
            failures.append((fname, repr(exc)))

        if (i + 1) % 100 == 0:
            elapsed = (time.time() - start) / 60.0
            atk_avg = float(np.mean(atk_running)) if atk_running else float("nan")
            best = float(np.max(best_running)) if best_running else float("nan")
            print(f"[{attack_tag}] {i + 1}/{num_images} | "
                  f"attacked={atk_avg:.4f} | best={best:.4f} | {elapsed:.1f}m")
        if (i + 1) % 250 == 0:
            torch.cuda.empty_cache()

    ckpt.flush()

    if failures:
        print(f"[{attack_key}] {len(failures)} image(s) failed:")
        for fn, exc in failures[:10]:
            print(f"    {fn}: {exc}")

    # --- HARD ASSERTS ------------------------------------------------------
    per_image = ckpt.buckets
    assert len(per_image) == num_images, (
        f"[{attack_key}] image count mismatch: processed {len(per_image)} "
        f"but expected {num_images} (failures={len(failures)})"
    )
    for iid, b in per_image.items():
        missing = [t for t in expected_tags if t not in b]
        assert not missing, f"[{attack_key}] img {iid} missing tags: {missing}"
    print(f"[{attack_key}] all {len(per_image)} images carry all "
          f"{len(expected_tags)} expected condition tags.")

    # --- Assemble + COCO eval ----------------------------------------------
    all_results = pc.assemble_results(
        per_image,
        defense_names=defense_names,
        attack_tags=[attack_tag],
        ensembles=ensembles,
        nms_iou=ENSEMBLE_NMS_IOU,
    )
    processed_ids = sorted(per_image.keys())
    assert len(processed_ids) == num_images, (
        f"[{attack_key}] processed_ids ({len(processed_ids)}) != num_images ({num_images})"
    )
    eval_stats = pc.evaluate_all_conditions(
        all_results, coco_gt, image_ids=processed_ids, output_dir=OUTPUT_DIR,
    )

    rows, clean_ap, atk_ap, attack_drop = ranked_table(
        eval_stats, attack_tag, defense_names, ensembles
    )

    # --- summary_{attack}.json ---------------------------------------------
    summary = {
        "model": model_name,
        "attack": attack_tag,
        "tier": ARGS.tier,
        "novel": ARGS.novel,
        "epsilon": EPSILON if attack_key != "patch" else None,
        "num_images": len(processed_ids),
        "clean_mAP": clean_ap,
        "clean_AP50": float(eval_stats["clean"][1]),
        "attacked_mAP": atk_ap,
        "attack_damage": attack_drop,
        "defense_names": defense_names,
        "ensembles": {k: v for k, v in ensembles.items()},
        "failures": failures,
        "defenses": {},
    }
    for dn, kind, def_ap, rec, rec_pct in rows:
        tag = f"{attack_tag}+{dn}"
        summary["defenses"][dn] = {
            "mAP": def_ap,
            "AP50": float(eval_stats[tag][1]),
            "recovery": rec,
            "recovery_pct": rec_pct,
            "kind": kind,
        }

    summary_path = os.path.join(OUTPUT_DIR, f"summary_{attack_key}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{attack_key}] wrote {summary_path}")


# ======================================================================
# 9. Drive all requested attacks SEQUENTIALLY in this one process.
# ======================================================================
def main():
    for attack_key in ARGS.attacks:
        run_attack(attack_key)
    pc.print_banner("Survey complete", width=70)


if __name__ == "__main__":
    main()
