#!/usr/bin/env python
"""
FGSM Phase 2 — Variant A: Classical Input Squeezing (YOLOv8x-worldv2)

Same defense pipeline as the Florence-2 version, but using YOLOv8x-worldv2
(73.7M params) for ~10-50x faster iteration.

Key differences from Florence-2 version:
  - No label mapping needed (YOLO outputs COCO categories directly)
  - Real confidence scores (no heuristic scoring)
  - Built-in NMS (no custom implementation)
  - FGSM uses detection confidence loss (not autoregressive cross-entropy)
  - Single forward pass per inference (not autoregressive token generation)

Defenses (same "Feature Squeezing" family — model-agnostic):
  | Defense              | Reference                         |
  |----------------------|-----------------------------------|
  | JPEG Compression q75 | Dziugaite et al., 2016            |
  | Gaussian Blur σ=1.0  | Xu et al., NDSS 2018              |
  | Median Filter 3×3    | Xu et al., NDSS 2018              |
  | Bit Depth Reduction  | Xu et al., NDSS 2018; Guo ICLR 18 |

Usage:
  conda activate vlm_ftune
  pip install ultralytics  # if not installed
  cd /path/to/Loki_CV
  python FGSM_Phase2_VariantA_YOLO.py
"""

# ============================================================
# 1. Setup and Imports
# ============================================================

import os
import json
import sys

# GPU ISOLATION -- Must happen BEFORE import torch
NUM_GPUS = 1

import subprocess
try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    gpu_free = []
    for line in smi.stdout.strip().split("\n"):
        idx, free = line.split(",")
        gpu_free.append((int(idx.strip()), int(free.strip())))
    gpu_free.sort(key=lambda x: x[1], reverse=True)
    selected = [str(g[0]) for g in gpu_free[:NUM_GPUS]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    print(f"GPU isolation: NUM_GPUS={NUM_GPUS}, "
          f"selected physical GPU(s): {selected}, "
          f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
except Exception as e:
    print(f"nvidia-smi query failed ({e}), using default CUDA_VISIBLE_DEVICES")

# Now safe to import torch
import torch
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
from tqdm.auto import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import warnings
import gc
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Check ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics is not installed.")
    print("Install it with: pip install ultralytics")
    sys.exit(1)

# Verify GPU isolation
n_visible = torch.cuda.device_count()
print(f"PyTorch sees {n_visible} GPU(s) (requested {NUM_GPUS})")
assert n_visible <= NUM_GPUS, (
    f"Expected at most {NUM_GPUS} GPU(s) but {n_visible} visible. "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
)

print("All core imports successful.")

# ============================================================
# 1.5 GPU Diagnostics & Device Selection
# ============================================================
print("=" * 70)
print(f"GPU DIAGNOSTICS (NUM_GPUS={NUM_GPUS})")
print("=" * 70)

try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    print(result.stdout)
except Exception as e:
    print(f"nvidia-smi failed: {e}")

print("-" * 70)
n_gpus = torch.cuda.device_count()
print(f"GPUs visible to PyTorch: {n_gpus}")

for i in range(n_gpus):
    free, total = torch.cuda.mem_get_info(i)
    props = torch.cuda.get_device_properties(i)
    used_pct = (total - free) / total * 100
    print(f"  GPU {i}: {props.name}")
    print(f"    Memory: {(total-free)/1024**2:.0f}MB used / {total/1024**2:.0f}MB total ({used_pct:.1f}% used)")
    print(f"    Free: {free/1024**3:.2f} GB")

device = torch.device("cuda:0")
torch_dtype = torch.float16

props = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info(0)
print(f"\n>> Primary device: {device} -> {props.name} ({free/1024**3:.2f} GB free)")
print(f"Device: {device}, Dtype: {torch_dtype}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print("=" * 70)

# ============================================================
# 2. Configuration
# ============================================================

# Dataset paths
IMAGE_DIR = "./Dataset/val2017"
ANN_FILE = "./Dataset/annotations/instances_val2017.json"

# Number of images
NUM_IMAGES = 5000

# FGSM epsilon values
EPSILONS = [0.003, 0.01, 0.03]

# Defenses
RUN_JPEG = True
RUN_GAUSSIAN_BLUR = True
RUN_MEDIAN_FILTER = True
RUN_BIT_DEPTH = True

# Parameters
JPEG_QUALITY = 75          # Dziugaite et al., 2016
BLUR_SIGMA = 1.0           # Xu et al., NDSS 2018
MEDIAN_KERNEL = 3          # Xu et al., NDSS 2018
BIT_DEPTH = 4              # Xu et al., NDSS 2018 (reduce 8-bit to 4-bit)

# YOLO settings
YOLO_MODEL = "yolov8x-worldv2.pt"
YOLO_IMGSZ = 640           # Model input size
YOLO_CONF = 0.001          # Low threshold for COCO mAP evaluation
YOLO_IOU_NMS = 0.5         # NMS IoU threshold

# Output
OUTPUT_DIR = "./results_phase2_variantA_yolo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Variant A: Classical Squeezing (YOLOv8x-worldv2)")
print(f"  Images: {NUM_IMAGES}, Epsilons: {EPSILONS}")
print(f"  Model: {YOLO_MODEL} (imgsz={YOLO_IMGSZ})")
print(f"  Defenses: JPEG={RUN_JPEG}, Blur={RUN_GAUSSIAN_BLUR}, Median={RUN_MEDIAN_FILTER}, BitDepth={RUN_BIT_DEPTH}")

# ============================================================
# 3. Load Model and Dataset
# ============================================================

print(f"\nUsing device: {device}")

# Load COCO ground truth first (needed for class names)
coco_gt = COCO(ANN_FILE)
cats = coco_gt.loadCats(coco_gt.getCatIds())
cats_sorted = sorted(cats, key=lambda x: x["id"])

# COCO class names and IDs in standard order
COCO_NAMES = [c["name"] for c in cats_sorted]   # 80 class names
COCO_IDS = [c["id"] for c in cats_sorted]        # 80 category IDs (1,2,3,...,90 with gaps)

# Mapping: YOLO class index i -> COCO category ID
YOLO_TO_COCO_ID = {i: cid for i, cid in enumerate(COCO_IDS)}

print(f"COCO categories loaded: {len(COCO_NAMES)}")

# Load YOLOv8x-worldv2
print(f"Loading {YOLO_MODEL}...")
model = YOLO(YOLO_MODEL)

# Set COCO classes as the vocabulary for open-vocabulary detection
model.set_classes(COCO_NAMES)
print(f"Model loaded and classes set ({len(COCO_NAMES)} COCO categories).")

# Get model input size and ensure model is on GPU
model.to(device)

# Image file list
files = sorted(os.listdir(IMAGE_DIR))
if NUM_IMAGES is not None:
    files = files[:NUM_IMAGES]

# Track which image IDs we evaluate (needed for COCOeval filtering)
evaluated_img_ids = sorted([int(os.path.splitext(f)[0]) for f in files])
print(f"Will process {len(files)} images (img_ids tracked for COCOeval).")

# ============================================================
# 4. Inference Functions
# ============================================================

def _parse_yolo_results(result):
    """Convert a single YOLO Results object to COCO-format detections."""
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


def run_inference(pil_img):
    """Run YOLOv8-World inference on a single image."""
    results = model.predict(
        pil_img, conf=YOLO_CONF, iou=YOLO_IOU_NMS,
        imgsz=YOLO_IMGSZ, verbose=False
    )
    return _parse_yolo_results(results[0])


def run_inference_batch(pil_imgs):
    """Run YOLOv8-World inference on a batch of images."""
    if not pil_imgs:
        return []
    results = model.predict(
        pil_imgs, conf=YOLO_CONF, iou=YOLO_IOU_NMS,
        imgsz=YOLO_IMGSZ, verbose=False
    )
    return [_parse_yolo_results(r) for r in results]

print("Inference pipeline ready (native COCO categories + real confidence scores).")

# ============================================================
# 5. FGSM Attack
# ============================================================


def _letterbox_image(pil_img, target_size=640, fill=(114, 114, 114)):
    """Resize image with letterbox padding (matching YOLO preprocessing)."""
    w, h = pil_img.size
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), fill)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))

    return canvas, scale, pad_left, pad_top, new_w, new_h


def _unletterbox_image(lb_np, orig_size, pad_left, pad_top, new_w, new_h):
    """Extract valid region from letterboxed array and resize to original."""
    cropped = lb_np[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    return Image.fromarray(cropped).resize(orig_size, Image.BICUBIC)


def fgsm_attack(pil_img, eps=0.01):
    """
    FGSM attack on YOLOv8-World.

    Strategy: Minimize total detection confidence by computing gradient
    of the sum of max class probabilities w.r.t. the input image.

    Unlike Florence-2 (which attacks autoregressive cross-entropy loss),
    this directly attacks the detection confidence output.
    """
    orig_size = pil_img.size  # (W, H)

    # Letterbox to model input size (matches YOLO's internal preprocessing)
    lb_img, scale, pad_left, pad_top, new_w, new_h = _letterbox_image(
        pil_img, YOLO_IMGSZ
    )

    # Convert to tensor [B, C, H, W] in [0, 1]
    img_np = np.array(lb_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(
        img_np.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    # Compute gradient
    adv_tensor = img_tensor.clone().detach().requires_grad_(True)

    # Forward pass through raw YOLO model
    preds = model.model(adv_tensor)
    if isinstance(preds, (list, tuple)):
        pred = preds[0]
    else:
        pred = preds

    # YOLOv8 output: [batch, 4+nc, num_anchors]
    # Extract class scores (everything after first 4 channels = bbox)
    nc = pred.shape[1] - 4
    cls_scores = pred[:, 4:, :]          # [1, nc, anchors]
    max_cls = cls_scores.max(dim=1)[0]   # [1, anchors] - max class prob per anchor

    # Loss = negative total confidence
    # Maximizing this loss = minimizing detection confidence = degrading detection
    loss = -max_cls.sum()
    loss.backward()

    grad_sign = adv_tensor.grad.sign()

    # Standard FGSM: x_adv = x + eps * sign(∇_x loss)
    perturbed = img_tensor.detach() + eps * grad_sign
    perturbed = torch.clamp(perturbed, 0.0, 1.0)

    # Convert back to PIL at original resolution
    adv_np = (perturbed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    adv_pil = _unletterbox_image(adv_np, orig_size, pad_left, pad_top, new_w, new_h)

    return adv_pil

print("FGSM attack function ready.")


def fgsm_attack_multi_eps(pil_img, epsilons):
    """
    FGSM for multiple epsilons in one shot.
    Computes gradient ONCE and applies different step sizes.
    Saves 1 forward + 1 backward per extra epsilon.
    """
    orig_size = pil_img.size

    # Letterbox
    lb_img, scale, pad_left, pad_top, new_w, new_h = _letterbox_image(
        pil_img, YOLO_IMGSZ
    )

    # To tensor
    img_np = np.array(lb_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(
        img_np.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    # Compute gradient (ONE forward + backward)
    adv_tensor = img_tensor.clone().detach().requires_grad_(True)
    preds = model.model(adv_tensor)
    if isinstance(preds, (list, tuple)):
        pred = preds[0]
    else:
        pred = preds

    nc = pred.shape[1] - 4
    cls_scores = pred[:, 4:, :]
    max_cls = cls_scores.max(dim=1)[0]
    loss = -max_cls.sum()
    loss.backward()

    grad_sign = adv_tensor.grad.sign()

    # Generate adversarial images for each epsilon (cheap tensor ops only)
    adv_images = {}
    for eps in epsilons:
        perturbed = img_tensor.detach() + eps * grad_sign
        perturbed = torch.clamp(perturbed, 0.0, 1.0)
        adv_np = (perturbed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        adv_pil = _unletterbox_image(adv_np, orig_size, pad_left, pad_top, new_w, new_h)
        adv_images[eps] = adv_pil

    return adv_images

print("Multi-epsilon FGSM attack ready (computes gradient once for all epsilons).")

# ============================================================
# 6. Defense Functions — Variant A: Classical Squeezing
# ============================================================

# Defense 1: JPEG Compression (Dziugaite et al., 2016)
def defend_jpeg(pil_img, quality=JPEG_QUALITY):
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

# Defense 2: Gaussian Blur (Xu et al., NDSS 2018)
def defend_gaussian_blur(pil_img, sigma=BLUR_SIGMA):
    return pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))

# Defense 3: Median Filter (Xu et al., NDSS 2018)
def defend_median_filter(pil_img, kernel=MEDIAN_KERNEL):
    return pil_img.filter(ImageFilter.MedianFilter(size=kernel))

# Defense 4: Bit Depth Reduction (Xu et al., NDSS 2018; Guo et al., ICLR 2018)
def defend_bit_depth(pil_img, bits=BIT_DEPTH):
    arr = np.array(pil_img)
    shift = 8 - bits
    arr = (arr >> shift) << shift
    return Image.fromarray(arr)

# Defense registry
DEFENSES = {}
if RUN_JPEG:
    DEFENSES["jpeg"] = defend_jpeg
if RUN_GAUSSIAN_BLUR:
    DEFENSES["gaussian_blur"] = defend_gaussian_blur
if RUN_MEDIAN_FILTER:
    DEFENSES["median_filter"] = defend_median_filter
if RUN_BIT_DEPTH:
    DEFENSES["bit_depth"] = defend_bit_depth

print(f"Defenses: {list(DEFENSES.keys())}")

# ============================================================
# 6.5 Sanity Check
# ============================================================
print("\nRunning sanity checks...")
errors = []

if not os.path.isdir(IMAGE_DIR):
    errors.append(f"IMAGE_DIR not found: {IMAGE_DIR}")
else:
    print(f"  [OK] Image directory: {len(os.listdir(IMAGE_DIR))} files")

if not os.path.isfile(ANN_FILE):
    errors.append(f"ANN_FILE not found: {ANN_FILE}")
else:
    print(f"  [OK] Annotation file")

try:
    test_img = Image.open(os.path.join(IMAGE_DIR, files[0])).convert("RGB")
    test_dets = run_inference(test_img)
    print(f"  [OK] Model inference: {len(test_dets)} detections (with real confidence scores)")
except Exception as e:
    errors.append(f"Inference failed: {e}")

try:
    test_adv = fgsm_attack(test_img, eps=0.01)
    print(f"  [OK] FGSM attack")
except Exception as e:
    errors.append(f"FGSM failed: {e}")

for dname, dfunc in DEFENSES.items():
    try:
        _ = dfunc(test_img)
        print(f"  [OK] {dname} defense")
    except Exception as e:
        errors.append(f"{dname} failed: {e}")

try:
    batch_dets = run_inference_batch([test_img, test_img])
    assert len(batch_dets) == 2, f"Expected 2 results, got {len(batch_dets)}"
    print(f"  [OK] Batched inference (batch=2): {len(batch_dets[0])}, {len(batch_dets[1])} detections")
except Exception as e:
    errors.append(f"Batched inference failed: {e}")

try:
    multi_adv = fgsm_attack_multi_eps(test_img, [0.01, 0.03])
    assert len(multi_adv) == 2, f"Expected 2 eps results, got {len(multi_adv)}"
    print(f"  [OK] Multi-epsilon FGSM attack ({list(multi_adv.keys())})")
except Exception as e:
    errors.append(f"Multi-eps FGSM failed: {e}")

if torch.cuda.is_available():
    mem = torch.cuda.memory_allocated(device) / 1024**3
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"  [OK] GPU: {mem:.2f}GB / {total_mem:.2f}GB")

n_conds = 1 + len(DEFENSES) + len(EPSILONS) * (1 + len(DEFENSES))
print()
if errors:
    print(f"FAILED -- {len(errors)} error(s):")
    for e in errors:
        print(f"  x {e}")
    sys.exit(1)
else:
    print(f"All checks passed!")
    print(f"  Images: {len(files)}, Defenses: {len(DEFENSES)}, Conditions: {n_conds}")
    print(f"  Estimated inference calls: ~{len(files) * n_conds}")

# ============================================================
# 7. COCO Evaluation Helper
# ============================================================

def evaluate_coco(results_list, tag="eval"):
    if not results_list:
        print(f"  [{tag}] No detections -- returning zero mAP.")
        return np.zeros(12)
    out_path = os.path.join(OUTPUT_DIR, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results_list, f)
    coco_dt = coco_gt.loadRes(out_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = evaluated_img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

print("COCO evaluation helper ready.")

# ============================================================
# 8. Main Evaluation Pipeline
# ============================================================

def run_full_evaluation():
    start_time = time.time()
    all_results = {}

    # Build condition list
    conditions = ["clean"]
    for dname in DEFENSES:
        conditions.append(f"clean+{dname}")
    for eps in EPSILONS:
        eps_tag = f"fgsm_eps{eps}"
        conditions.append(eps_tag)
        for dname in DEFENSES:
            conditions.append(f"{eps_tag}+{dname}")

    for cond in conditions:
        all_results[cond] = []

    print(f"Conditions to evaluate: {len(conditions)}")
    for c in conditions:
        print(f"  - {c}")
    print()

    defense_names = list(DEFENSES.keys())
    defense_funcs = list(DEFENSES.values())

    for fname in tqdm(files, desc="Processing images"):
        img_id = int(os.path.splitext(fname)[0])
        img_path = os.path.join(IMAGE_DIR, fname)
        pil_img = Image.open(img_path).convert("RGB")

        # ---- Prepare all image variants for this image ----
        batch_imgs = []
        batch_tags = []

        # Clean baseline
        batch_imgs.append(pil_img)
        batch_tags.append("clean")

        # Clean + each defense
        for dname, dfunc in zip(defense_names, defense_funcs):
            batch_imgs.append(dfunc(pil_img))
            batch_tags.append(f"clean+{dname}")

        # FGSM attacks — compute gradient ONCE for all epsilons
        adv_images = fgsm_attack_multi_eps(pil_img, EPSILONS)

        for eps in EPSILONS:
            eps_tag = f"fgsm_eps{eps}"
            adv_img = adv_images[eps]

            # Attacked (no defense)
            batch_imgs.append(adv_img)
            batch_tags.append(eps_tag)

            # Attacked + each defense
            for dname, dfunc in zip(defense_names, defense_funcs):
                batch_imgs.append(dfunc(adv_img))
                batch_tags.append(f"{eps_tag}+{dname}")

        # ---- Run inference on all variants ----
        # YOLO is fast enough to process all at once
        all_dets = run_inference_batch(batch_imgs)

        for tag, dets in zip(batch_tags, all_dets):
            for d in dets:
                d["image_id"] = img_id
            all_results[tag].extend(dets)

        # Free memory periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\nAll inference done in {elapsed/60:.1f} minutes.")
    return all_results

all_results = run_full_evaluation()

# ============================================================
# 9. COCO Evaluation for All Conditions
# ============================================================

eval_stats = {}
for tag, results_list in all_results.items():
    print(f"\n{'='*60}")
    print(f"Evaluating: {tag} ({len(results_list)} detections)")
    print(f"{'='*60}")
    stats = evaluate_coco(results_list, tag=tag)
    eval_stats[tag] = stats
print("\nAll evaluations complete.")

# ============================================================
# 10. Results Summary Table
# ============================================================

clean_ap = eval_stats["clean"][0]
clean_ap50 = eval_stats["clean"][1]

all_defense_names = list(DEFENSES.keys())

print("=" * 90)
print(f"{'FGSM ATTACK & DEFENSE RESULTS (VARIANT A — YOLOv8x-worldv2)':^90}")
print("=" * 90)
print(f"\nModel: {YOLO_MODEL} | Images: {NUM_IMAGES}")
print(f"Clean Baseline:  mAP = {clean_ap:.4f},  AP50 = {clean_ap50:.4f}")
print()

print("-" * 60)
print("TABLE 1: Defense Cost (on CLEAN images, no attack)")
print("-" * 60)
print(f"  {'Condition':<25} {'mAP':>8} {'AP50':>8} {'mAP Drop':>10}")
print(f"  {'clean (baseline)':<25} {clean_ap:>8.4f} {clean_ap50:>8.4f} {'---':>10}")
for dname in all_defense_names:
    tag = f"clean+{dname}"
    if tag in eval_stats:
        ap, ap50 = eval_stats[tag][0], eval_stats[tag][1]
        drop = clean_ap - ap
        print(f"  {'clean + ' + dname:<25} {ap:>8.4f} {ap50:>8.4f} {drop:>+10.4f}")
print()

print("-" * 90)
print("TABLE 2: Attack Impact & Defense Recovery")
print("-" * 90)
for eps in EPSILONS:
    eps_tag = f"fgsm_eps{eps}"
    atk_ap = eval_stats[eps_tag][0]
    atk_drop = clean_ap - atk_ap
    print(f"\n  FGSM eps={eps}:")
    print(f"    {'Condition':<30} {'mAP':>8} {'AP50':>8} {'Recovery':>10} {'Recovery%':>12}")
    print(f"    {'attacked (no defense)':<30} {atk_ap:>8.4f} {eval_stats[eps_tag][1]:>8.4f} {'---':>10} {'---':>12}")
    for dname in all_defense_names:
        def_tag = f"{eps_tag}+{dname}"
        if def_tag in eval_stats:
            def_ap, def_ap50 = eval_stats[def_tag][0], eval_stats[def_tag][1]
            recovery = def_ap - atk_ap
            recovery_pct = (recovery / atk_drop * 100) if atk_drop > 0 else 0
            print(f"    {'attacked + ' + dname:<30} {def_ap:>8.4f} {def_ap50:>8.4f} {recovery:>+10.4f} {recovery_pct:>11.1f}%")

print("\n" + "=" * 90)

# Save summary JSON
summary = {
    "model": YOLO_MODEL,
    "num_images": NUM_IMAGES,
    "clean_mAP": float(clean_ap),
    "clean_AP50": float(clean_ap50),
    "defense_cost": {},
    "attack_results": {},
}
for dname in all_defense_names:
    tag = f"clean+{dname}"
    if tag in eval_stats:
        summary["defense_cost"][dname] = {
            "mAP": float(eval_stats[tag][0]),
            "mAP_drop": float(clean_ap - eval_stats[tag][0]),
        }
for eps in EPSILONS:
    eps_tag = f"fgsm_eps{eps}"
    atk_ap = eval_stats[eps_tag][0]
    atk_drop = clean_ap - atk_ap
    entry = {"attacked_mAP": float(atk_ap), "defenses": {}}
    for dname in all_defense_names:
        def_tag = f"{eps_tag}+{dname}"
        if def_tag in eval_stats:
            def_ap = eval_stats[def_tag][0]
            recovery = def_ap - atk_ap
            recovery_pct = (recovery / atk_drop * 100) if atk_drop > 0 else 0
            entry["defenses"][dname] = {
                "mAP": float(def_ap), "recovery": float(recovery),
                "recovery_pct": float(recovery_pct),
            }
    summary["attack_results"][str(eps)] = entry

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to {OUTPUT_DIR}/summary.json")

# ============================================================
# 11. Visualization
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
atk_maps = [eval_stats[f"fgsm_eps{e}"][0] for e in EPSILONS]
ax.plot(EPSILONS, atk_maps, 'r-o', linewidth=2, markersize=8, label="Attacked (no defense)")

colors = ['blue', 'green', 'orange', 'purple', 'brown']
for i, dname in enumerate(all_defense_names):
    def_maps = [eval_stats.get(f"fgsm_eps{e}+{dname}", np.zeros(12))[0] for e in EPSILONS]
    ax.plot(EPSILONS, def_maps, '-s', color=colors[i % len(colors)],
            linewidth=2, markersize=7, label=f"+ {dname}")

ax.axhline(y=clean_ap, color='gray', linestyle='--', linewidth=1.5, label=f"Clean ({clean_ap:.3f})")
ax.set_xlabel("FGSM Epsilon", fontsize=12)
ax.set_ylabel("mAP", fontsize=12)
ax.set_title("FGSM Attack: mAP vs Epsilon (YOLOv8x-worldv2)", fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
clean_maps = [eval_stats.get(f"clean+{d}", np.zeros(12))[0] for d in all_defense_names]
clean_costs = [clean_ap - m for m in clean_maps]
bars = ax.bar(all_defense_names, clean_maps, color=colors[:len(all_defense_names)], alpha=0.8)
ax.axhline(y=clean_ap, color='gray', linestyle='--', linewidth=1.5, label=f"Clean baseline ({clean_ap:.3f})")
for bar, cost in zip(bars, clean_costs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"-{cost:.3f}", ha='center', va='bottom', fontsize=9)
ax.set_ylabel("mAP", fontsize=12)
ax.set_title("Defense Cost on Clean Images (YOLOv8x-worldv2)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "results_plot.png"), dpi=150, bbox_inches='tight')
print(f"Plot saved to {OUTPUT_DIR}/results_plot.png")

# Visual comparison of defenses on a sample image
sample_fname = files[0]
sample_img = Image.open(os.path.join(IMAGE_DIR, sample_fname)).convert("RGB")
sample_adv = fgsm_attack(sample_img, eps=0.03)

std_names = list(DEFENSES.keys())
ncols = 3 + len(std_names)
fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

axes[0].imshow(sample_img); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(sample_adv); axes[1].set_title("FGSM (eps=0.03)"); axes[1].axis("off")

diff = np.abs(np.array(sample_img).astype(float) - np.array(sample_adv).astype(float))
diff = np.clip(diff * 10, 0, 255).astype(np.uint8)
axes[2].imshow(diff); axes[2].set_title("Difference (x10)"); axes[2].axis("off")

for i, dname in enumerate(std_names):
    defended = DEFENSES[dname](sample_adv)
    axes[3 + i].imshow(defended); axes[3 + i].set_title(f"+ {dname}"); axes[3 + i].axis("off")

plt.suptitle("FGSM Attack and Defense Visual Comparison (YOLOv8x-worldv2)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "visual_comparison.png"), dpi=150, bbox_inches='tight')
print(f"Visual comparison saved to {OUTPUT_DIR}/visual_comparison.png")

# ============================================================
# 12. Net Gain Analysis
# ============================================================

print("=" * 80)
print(f"{'NET GAIN ANALYSIS (VARIANT A — YOLOv8x-worldv2)':^80}")
print("=" * 80)
print("\nNet Gain = defended_mAP - max(clean+defense_mAP, attacked_mAP)")
print("Positive = defense is helpful. Negative = defense makes things worse.\n")

for eps in EPSILONS:
    eps_tag = f"fgsm_eps{eps}"
    atk_ap = eval_stats[eps_tag][0]
    print(f"  FGSM eps={eps} (attacked mAP = {atk_ap:.4f}):")
    for dname in all_defense_names:
        def_tag = f"{eps_tag}+{dname}"
        clean_def_tag = f"clean+{dname}"
        if def_tag in eval_stats and clean_def_tag in eval_stats:
            def_ap = eval_stats[def_tag][0]
            clean_def_ap = eval_stats[clean_def_tag][0]
            floor = max(atk_ap, clean_def_ap)
            net_gain = def_ap - floor
            verdict = "HELPFUL" if net_gain > 0 else "NOT HELPFUL"
            print(f"    {dname:<15} defended_mAP={def_ap:.4f}  "
                  f"floor={floor:.4f}  net_gain={net_gain:+.4f}  [{verdict}]")
    print()

print("=" * 80)
print("DONE. All results saved to:", OUTPUT_DIR)
