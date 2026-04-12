#!/usr/bin/env python
"""
PGD Phase 3 — Iterative Adversarial Attacks on YOLOv8x-worldv2

This script evaluates the best defenses identified from FGSM Phase 2 against 
a much stronger PGD (Projected Gradient Descent) iterative attack.
"""

import os
import json
import sys

# GPU ISOLATION
NUM_GPUS = 1

import subprocess
import torch
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
from tqdm.auto import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from skimage.restoration import denoise_tv_chambolle
import time
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from ultralytics import YOLO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. Configuration
# ============================================================

IMAGE_DIR = "./Dataset/val2017"
ANN_FILE = "./Dataset/annotations/instances_val2017.json"

# Set to 100 images for a very fast preliminary readout.
NUM_IMAGES = 100

EPSILONS = [0.003, 0.01, 0.03]
PGD_ITERS = 10

# Top Performing Defenses from Phase 2
RUN_BLUR_TVM = True
RUN_MEDIAN = True
RUN_JPEG = True
RUN_TVM = True
RUN_JPEG_TVM_NLM = True # Retaining for checking collapse

JPEG_QUALITY = 75
BLUR_SIGMA = 1.0
TVM_WEIGHT = 0.05
NLM_H = 6

YOLO_MODEL = "yolov8x-worldv2.pt"
YOLO_IMGSZ = 640
YOLO_CONF = 0.001
YOLO_IOU_NMS = 0.5

OUTPUT_DIR = "./results_phase3_pgd_yolo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Phase 3: PGD Attack (iters={PGD_ITERS})")
print(f"Images: {NUM_IMAGES}, Epsilons: {EPSILONS}")

# ============================================================
# 3. Load COCO & YOLO
# ============================================================

coco_gt = COCO(ANN_FILE)
cats = coco_gt.loadCats(coco_gt.getCatIds())
cats_sorted = sorted(cats, key=lambda x: x["id"])

COCO_NAMES = [c["name"] for c in cats_sorted]
COCO_IDS = [c["id"] for c in cats_sorted]
YOLO_TO_COCO_ID = {i: cid for i, cid in enumerate(COCO_IDS)}

model = YOLO(YOLO_MODEL)
model.set_classes(COCO_NAMES)
model.to(device)

files = sorted(os.listdir(IMAGE_DIR))
if NUM_IMAGES is not None:
    files = files[:NUM_IMAGES]
evaluated_img_ids = sorted([int(os.path.splitext(f)[0]) for f in files])

# ============================================================
# 4. Inference
# ============================================================

def _parse_yolo_results(result):
    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy[i]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        cls_idx = int(clss[i])
        if cls_idx not in YOLO_TO_COCO_ID:
            continue
        detections.append({
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "category_id": YOLO_TO_COCO_ID[cls_idx],
            "score": float(confs[i]),
        })
    return detections

def run_inference_batch(pil_imgs):
    if not pil_imgs: return []
    results = model.predict(pil_imgs, conf=YOLO_CONF, iou=YOLO_IOU_NMS, imgsz=YOLO_IMGSZ, verbose=False)
    return [_parse_yolo_results(r) for r in results]

# ============================================================
# 5. PGD Attack
# ============================================================

def _letterbox_image(pil_img, target_size=640, fill=(114, 114, 114)):
    w, h = pil_img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), fill)
    pad_left, pad_top = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, scale, pad_left, pad_top, new_w, new_h

def _unletterbox_image(lb_np, orig_size, pad_left, pad_top, new_w, new_h):
    cropped = lb_np[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    return Image.fromarray(cropped).resize(orig_size, Image.BICUBIC)

def pgd_attack(pil_img, eps, iters=10):
    alpha = eps / 4.0
    orig_size = pil_img.size
    lb_img, scale, pad_left, pad_top, new_w, new_h = _letterbox_image(pil_img, YOLO_IMGSZ)

    img_np = np.array(lb_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Random Start
    adv_tensor = img_tensor.clone().detach() + torch.empty_like(img_tensor).uniform_(-eps, eps)
    adv_tensor = torch.clamp(adv_tensor, 0.0, 1.0)

    for _ in range(iters):
        adv_tensor.requires_grad_(True)
        preds = model.model(adv_tensor)
        pred = preds[0] if isinstance(preds, (list, tuple)) else preds
        
        cls_scores = pred[:, 4:, :]
        loss = -cls_scores.max(dim=1)[0].sum()
        
        if adv_tensor.grad is not None:
            adv_tensor.grad.zero_()
        loss.backward()

        grad_sign = adv_tensor.grad.sign()
        
        with torch.no_grad():
            adv_tensor = adv_tensor + alpha * grad_sign
            delta = torch.clamp(adv_tensor - img_tensor, min=-eps, max=eps)
            adv_tensor = torch.clamp(img_tensor + delta, 0.0, 1.0).detach()

    adv_np = (adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return _unletterbox_image(adv_np, orig_size, pad_left, pad_top, new_w, new_h)

def pgd_attack_multi_eps(pil_img, epsilons, iters=PGD_ITERS):
    return {eps: pgd_attack(pil_img, eps, iters) for eps in epsilons}

# ============================================================
# 6. Defenses
# ============================================================

def _jpeg(pil_img, quality=JPEG_QUALITY):
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def _gaussian_blur(pil_img, sigma=BLUR_SIGMA):
    return pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))

def _median(pil_img):
    return pil_img.filter(ImageFilter.MedianFilter(size=3))

def _tvm(pil_img, weight=TVM_WEIGHT):
    arr = np.array(pil_img).astype(np.float64) / 255.0
    denoised = denoise_tv_chambolle(arr, weight=weight, channel_axis=-1)
    return Image.fromarray((np.clip(denoised, 0, 1) * 255).astype(np.uint8))

def _nlm(pil_img, h=NLM_H):
    arr = np.array(pil_img)
    return Image.fromarray(cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21))

DEFENSES = {}
if RUN_TVM:
    DEFENSES["tvm"] = lambda img: _tvm(img)
if RUN_JPEG:
    DEFENSES["jpeg"] = lambda img: _jpeg(img)
if RUN_MEDIAN:
    DEFENSES["median_filter"] = lambda img: _median(img)
if RUN_BLUR_TVM:
    DEFENSES["blur_tvm"] = lambda img: _tvm(_gaussian_blur(img))
if RUN_JPEG_TVM_NLM:
    DEFENSES["jpeg_tvm_nlm"] = lambda img: _nlm(_tvm(_jpeg(img)))

# ============================================================
# 8. Evaluation Pipeline
# ============================================================

all_results = {"clean": []}
for d in DEFENSES: all_results[f"clean+{d}"] = []
for eps in EPSILONS:
    all_results[f"pgd_eps{eps}"] = []
    for d in DEFENSES: all_results[f"pgd_eps{eps}+{d}"] = []

defense_names, defense_funcs = list(DEFENSES.keys()), list(DEFENSES.values())

start_time = time.time()
for fname in tqdm(files, desc="PGD Evaluation"):
    img_id = int(os.path.splitext(fname)[0])
    img_path = os.path.join(IMAGE_DIR, fname)
    pil_img = Image.open(img_path).convert("RGB")

    batch_imgs, batch_tags = [pil_img], ["clean"]
    for dname, dfunc in zip(defense_names, defense_funcs):
        batch_imgs.append(dfunc(pil_img))
        batch_tags.append(f"clean+{dname}")

    adv_images = pgd_attack_multi_eps(pil_img, EPSILONS)

    for eps in EPSILONS:
        adv_img = adv_images[eps]
        batch_imgs.append(adv_img)
        batch_tags.append(f"pgd_eps{eps}")

        for dname, dfunc in zip(defense_names, defense_funcs):
            batch_imgs.append(dfunc(adv_img))
            batch_tags.append(f"pgd_eps{eps}+{dname}")

    all_dets = run_inference_batch(batch_imgs)
    for tag, dets in zip(batch_tags, all_dets):
        for d in dets: d["image_id"] = img_id
        all_results[tag].extend(dets)

print(f"Inference complete in {(time.time() - start_time)/60:.1f} mins.")

# ============================================================
# 9. COCO Eval
# ============================================================

eval_stats = {}
for tag, results_list in all_results.items():
    if not results_list:
        eval_stats[tag] = np.zeros(12)
        continue
    out_path = os.path.join(OUTPUT_DIR, f"{tag}.json")
    with open(out_path, "w") as f: json.dump(results_list, f)
    coco_dt = coco_gt.loadRes(out_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = evaluated_img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    eval_stats[tag] = coco_eval.stats

# ============================================================
# 10. Summary
# ============================================================

clean_ap, clean_ap50 = eval_stats["clean"][0], eval_stats["clean"][1]
summary = {"model": YOLO_MODEL, "num_images": NUM_IMAGES, "clean_mAP": float(clean_ap), "attack_results": {}}

for eps in EPSILONS:
    eps_tag = f"pgd_eps{eps}"
    atk_ap = eval_stats[eps_tag][0]
    entry = {"attacked_mAP": float(atk_ap), "defenses": {}}
    for dname in defense_names:
        def_ap = eval_stats[f"{eps_tag}+{dname}"][0]
        entry["defenses"][dname] = {"mAP": float(def_ap), "recovery": float(def_ap - atk_ap)}
    summary["attack_results"][str(eps)] = entry
    
with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== PGD NET GAIN ANALYSIS ===")
for eps in EPSILONS:
    atk_ap = eval_stats[f"pgd_eps{eps}"][0]
    print(f"\nPGD eps={eps} (attacked mAP = {atk_ap:.4f}):")
    for dname in defense_names:
        def_ap = eval_stats[f"pgd_eps{eps}+{dname}"][0]
        floor = max(atk_ap, eval_stats[f"clean+{dname}"][0])
        print(f"  {dname:<20} def_mAP={def_ap:.4f} net_gain={def_ap - floor:+.4f}")
