#!/usr/bin/env python
"""
FGSM Phase 3 — Florence-2 Hybrid Novel Evaluation
"""

# ============================================================
# 1. Setup and Imports
# ============================================================

import os
import json

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

# Now safe to import torch -- it will only see the selected GPU(s)
import torch
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms
from skimage.restoration import denoise_tv_chambolle
import GPUtil
import time
import warnings
import gc
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Verify GPU isolation
n_visible = torch.cuda.device_count()
print(f"PyTorch sees {n_visible} GPU(s) (requested {NUM_GPUS})")
assert n_visible <= NUM_GPUS, (
    f"Expected at most {NUM_GPUS} GPU(s) but {n_visible} visible. "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
)

# DiffPure REMOVED -- Nie et al. 2022 tested only on classification, not OD
# SmoothVLM REMOVED -- Sun et al. 2024 tested only on text generation, not OD

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
NUM_IMAGES = 500

# FGSM epsilon values
EPSILONS = [0.03]

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

# NMS threshold
NMS_IOU_THRESHOLD = 0.5

# Batch size for inference (higher = faster, more VRAM)
BATCH_SIZE = 10

# Output
OUTPUT_DIR = "./results_phase2_variantY"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Variant Y: Hybrid Occlusion")
print(f"  Images: {NUM_IMAGES}, Epsilons: {EPSILONS}")
print(f"  Defenses: JPEG={RUN_JPEG}, Blur={RUN_GAUSSIAN_BLUR}, Median={RUN_MEDIAN_FILTER}, BitDepth={RUN_BIT_DEPTH}")

# ============================================================
# 3. Load Model and Dataset
# ============================================================

# Device already selected in GPU Diagnostics cell above
print(f"Using device: {device}, dtype: {torch_dtype}")

model_name = "microsoft/Florence-2-base"
revision = "refs/pr/26"

print("Loading Florence-2-Base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, revision=revision,
    torch_dtype=torch_dtype, trust_remote_code=True
).to(device)
print("Model loaded.")

processor = AutoProcessor.from_pretrained(
    model_name, revision=revision, trust_remote_code=True
)

coco_gt = COCO(ANN_FILE)
categories = coco_gt.loadCats(coco_gt.getCatIds())
category_mapping = {c["name"]: c["id"] for c in categories}
print(f"COCO categories loaded: {len(category_mapping)}")

IMG_MEAN = torch.tensor(processor.image_processor.image_mean, device=device, dtype=torch_dtype).view(1, 3, 1, 1)
IMG_STD = torch.tensor(processor.image_processor.image_std, device=device, dtype=torch_dtype).view(1, 3, 1, 1)

files = sorted(os.listdir(IMAGE_DIR))
if NUM_IMAGES is not None:
    files = files[:NUM_IMAGES]
# Track which image IDs we evaluate (needed for COCOeval filtering)
evaluated_img_ids = sorted([int(os.path.splitext(f)[0]) for f in files])
print(f"Will process {len(files)} images (img_ids tracked for COCOeval).")

# ============================================================
# 4. Core Utilities: NMS and Inference
# ============================================================

# Florence-2 -> COCO Label Mapping
# Florence-2 outputs many labels that don't match COCO names.
# Without this mapping, ~55% of detections are silently dropped.

FLORENCE_TO_COCO = {
    # Persons
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "baby": "person", "kid": "person", "player": "person",
    "pedestrian": "person", "human": "person", "skier": "person",
    "snowboarder": "person", "surfer": "person", "rider": "person",
    # Vehicles
    "automobile": "car", "van": "car", "sedan": "car", "suv": "car",
    "taxi": "car", "minivan": "car",
    "motor bike": "motorcycle", "motorbike": "motorcycle",
    "aeroplane": "airplane", "aircraft": "airplane", "jet": "airplane",
    "lorry": "truck", "pickup truck": "truck",
    # Electronics
    "television": "tv", "tv set": "tv", "monitor": "tv", "screen": "tv",
    "television set": "tv",
    "mobile phone": "cell phone", "cellphone": "cell phone",
    "smartphone": "cell phone", "phone": "cell phone",
    "computer keyboard": "keyboard",
    "computer mouse": "mouse",
    "notebook computer": "laptop", "notebook": "laptop",
    # Furniture
    "studio couch": "couch", "sofa": "couch", "settee": "couch",
    "kitchen & dining room table": "dining table", "table": "dining table",
    "desk": "dining table",
    "swivel chair": "chair", "armchair": "chair", "stool": "chair",
    # Animals
    "puppy": "dog", "kitten": "cat",
    # Sports
    "ski": "skis", "ski pole": "skis",
    "racket": "tennis racket",
    "ball": "sports ball", "football": "sports ball",
    "soccer ball": "sports ball", "baseball": "sports ball",
    "basketball": "sports ball", "tennis ball": "sports ball",
    # Objects
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
    """Map Florence-2 label to COCO name. Try exact match first, then mapping."""
    if label in category_mapping:
        return label
    mapped = FLORENCE_TO_COCO.get(label)
    if mapped and mapped in category_mapping:
        return mapped
    lower = label.lower()
    if lower in category_mapping:
        return lower
    mapped_lower = FLORENCE_TO_COCO.get(lower)
    if mapped_lower and mapped_lower in category_mapping:
        return mapped_lower
    return None


def _compute_score(box, img_w, img_h):
    """
    Heuristic confidence score based on box geometry.
    COCO mAP needs varying scores for meaningful precision-recall curves.
    Applied identically to ALL conditions (clean, attacked, defended) for fairness.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    img_area = img_w * img_h
    box_area = w * h
    area_ratio = min(box_area / img_area, 0.5) if img_area > 0 else 0
    box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2
    img_cx, img_cy = img_w / 2, img_h / 2
    center_dist = np.sqrt(((box_cx - img_cx) / img_w) ** 2 +
                          ((box_cy - img_cy) / img_h) ** 2)
    score = 0.6 + 0.2 * area_ratio + 0.15 * (1 - center_dist)
    return min(0.98, max(0.6, score))


# ============================================================
# NMS
# ============================================================

def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def non_max_suppression(boxes, labels, scores, iou_thr=0.5):
    if not boxes:
        return [], [], []
    boxes = np.array(boxes)
    idxs = np.argsort(scores)[::-1]
    keep, keep_labels, keep_scores = [], [], []
    for i in idxs:
        suppress = False
        for j in keep:
            if labels[i] == labels[j] and box_iou(boxes[i], boxes[j]) > iou_thr:
                suppress = True
                break
        if not suppress:
            keep.append(i)
            keep_labels.append(labels[i])
            keep_scores.append(scores[i])
    return boxes[keep].tolist(), keep_labels, keep_scores


# ============================================================
# Inference (universal for all conditions)
# ============================================================

def run_inference(pil_img):
    """Run Florence-2 OD with label mapping and heuristic scores."""
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
    scores = [_compute_score(box, img_w, img_h) for box in bboxes]
    kept_boxes, kept_labels, kept_scores = non_max_suppression(
        bboxes, labels, scores, iou_thr=NMS_IOU_THRESHOLD
    )
    results = []
    for box, label, score in zip(kept_boxes, kept_labels, kept_scores):
        mapped = _map_label(label)
        if mapped is None:
            continue
        cid = category_mapping[mapped]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        results.append({"bbox": [x1, y1, w, h], "category_id": cid, "score": score})
    return results

print("Inference pipeline ready (with label mapping + heuristic scores).")

# ============================================================
# Batched Inference (processes multiple images in one generate call)
# ============================================================

def run_inference_batch(pil_imgs):
    """Run Florence-2 OD on a batch of images. Returns list of results per image."""
    if not pil_imgs:
        return []

    img_sizes = [img.size for img in pil_imgs]  # (w, h) per image

    with torch.no_grad():
        texts = ["<OD>"] * len(pil_imgs)
        inputs = processor(text=texts, images=pil_imgs, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        gen_kwargs = dict(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=512, num_beams=5, do_sample=False,
            repetition_penalty=1.8, length_penalty=1.0,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        gen_ids = model.generate(**gen_kwargs)
        texts_out = processor.batch_decode(gen_ids, skip_special_tokens=False)

    batch_results = []
    for txt, (img_w, img_h) in zip(texts_out, img_sizes):
        parsed = processor.post_process_generation(
            txt, task="<OD>", image_size=(img_w, img_h)
        ) or {}

        od = parsed.get("<OD>", {})
        bboxes, labels = od.get("bboxes", []), od.get("labels", [])
        scores = [_compute_score(box, img_w, img_h) for box in bboxes]
        kept_boxes, kept_labels, kept_scores = non_max_suppression(
            bboxes, labels, scores, iou_thr=NMS_IOU_THRESHOLD
        )
        results = []
        for box, label, score in zip(kept_boxes, kept_labels, kept_scores):
            mapped = _map_label(label)
            if mapped is None:
                continue
            cid = category_mapping[mapped]
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            results.append({"bbox": [x1, y1, w, h], "category_id": cid, "score": score})
        batch_results.append(results)

    return batch_results

print("Batched inference pipeline ready (BATCH_SIZE={}).".format(BATCH_SIZE))

# ============================================================
# 5. FGSM Attack
# ============================================================

def fgsm_attack(pil_img, eps=0.01):
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
    # Ensure labels are [1, seq_len] -- beam search may return extra rows
    target_ids = target_ids[:1, :].contiguous()
    if target_ids.size(1) > 512:
        target_ids = target_ids[:, :512].contiguous()

    pixel_values_adv = pixel_values.clone().detach().requires_grad_(True)
    outputs = model(input_ids=input_ids, pixel_values=pixel_values_adv, labels=target_ids)
    outputs.loss.backward()

    grad_sign = pixel_values_adv.grad.sign()
    adv_pixel_values = pixel_values.detach() + eps * grad_sign
    adv_pixel_values = torch.clamp(adv_pixel_values, -2.5, 2.5)

    # Denormalize: pixel_values shape is [1, C, H, W]
    mean = IMG_MEAN.squeeze(0)   # [C, 1, 1]
    std  = IMG_STD.squeeze(0)    # [C, 1, 1]
    adv_denorm = adv_pixel_values.squeeze(0) * std + mean
    adv_denorm = torch.clamp(adv_denorm, 0.0, 1.0)
    adv_np = (adv_denorm.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    if adv_pil.size != orig_size:
        adv_pil = adv_pil.resize(orig_size, Image.BICUBIC)
    return adv_pil

print("FGSM attack function ready.")

def fgsm_attack_multi_eps(pil_img, epsilons):
    """
    FGSM for multiple epsilons in one shot.
    The target_ids and gradient are identical for all epsilons (they depend
    only on the clean image), so we compute them ONCE and vary the step size.
    Saves 2 expensive model calls per extra epsilon.
    """
    orig_size = pil_img.size
    inputs = processor(text="<OD>", images=pil_img, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype)

    # 1) Get target_ids (model's own clean prediction) — ONE generate call
    with torch.no_grad():
        target_ids = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=512, num_beams=5, do_sample=False,
            repetition_penalty=1.8, length_penalty=1.0,
        )
    target_ids = target_ids[:1, :].contiguous()
    if target_ids.size(1) > 512:
        target_ids = target_ids[:, :512].contiguous()

    # 2) Compute gradient sign — ONE forward + backward
    pixel_values_adv = pixel_values.clone().detach().requires_grad_(True)
    outputs = model(input_ids=input_ids, pixel_values=pixel_values_adv, labels=target_ids)
    outputs.loss.backward()
    grad_sign = pixel_values_adv.grad.sign()

    # 3) Generate adversarial images for each epsilon — cheap tensor ops only
    mean = IMG_MEAN.squeeze(0)
    std = IMG_STD.squeeze(0)
    adv_images = {}
    for eps in epsilons:
        adv_pv = pixel_values.detach() + eps * grad_sign
        adv_pv = torch.clamp(adv_pv, -2.5, 2.5)
        adv_denorm = adv_pv.squeeze(0) * std + mean
        adv_denorm = torch.clamp(adv_denorm, 0.0, 1.0)
        adv_np = (adv_denorm.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        adv_pil = Image.fromarray(adv_np)
        if adv_pil.size != orig_size:
            adv_pil = adv_pil.resize(orig_size, Image.BICUBIC)
        adv_images[eps] = adv_pil

    return adv_images

print("Multi-epsilon FGSM attack ready (computes gradient once for all epsilons).")

# ============================================================
# 6. Defense Functions — Phase 3 Hybrid
# ============================================================

def _jpeg(pil_img, quality=75):
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def _median(pil_img, kernel=3):
    return pil_img.filter(ImageFilter.MedianFilter(size=kernel))

def _tvm(pil_img, weight=0.05):
    arr = np.array(pil_img).astype(np.float64) / 255.0
    denoised = denoise_tv_chambolle(arr, weight=weight, channel_axis=-1)
    return Image.fromarray((np.clip(denoised, 0, 1) * 255).astype(np.uint8))

def _random_cutout(pil_img, cutout_prob=0.15, block_size=16):
    arr = np.array(pil_img)
    h, w, c = arr.shape
    num_blocks_y = h // block_size
    num_blocks_x = w // block_size
    
    mask = np.random.rand(num_blocks_y, num_blocks_x) > cutout_prob
    mask_upsampled = mask.repeat(block_size, axis=0).repeat(block_size, axis=1)
    
    full_mask = np.ones((h, w), dtype=bool)
    full_mask[:mask_upsampled.shape[0], :mask_upsampled.shape[1]] = mask_upsampled
    
    arr[~full_mask] = 0
    return Image.fromarray(arr)

DEFENSES = {
    "jpeg_median_tvm_cutout": lambda img: _random_cutout(_tvm(_median(_jpeg(img)))),
    "jpeg_median_cutout": lambda img: _random_cutout(_median(_jpeg(img))),
    "jpeg_tvm_cutout": lambda img: _random_cutout(_tvm(_jpeg(img))),
    "median_tvm_cutout": lambda img: _random_cutout(_tvm(_median(img))),
    "jpeg_median": lambda img: _median(_jpeg(img)),
    "tvm_cutout": lambda img: _random_cutout(_tvm(img))
}

print(f"Defenses: {list(DEFENSES.keys())}")

# ============================================================
# 6.5 Sanity Check
# ============================================================
print("Running sanity checks...")
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
    print(f"  [OK] Model inference: {len(test_dets)} detections")
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
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"  [OK] GPU: {mem:.2f}GB / {total:.2f}GB")

n_conds = 1 + len(DEFENSES) + len(EPSILONS) * (1 + len(DEFENSES))
print()
if errors:
    print(f"FAILED -- {len(errors)} error(s):")
    for e in errors:
        print(f"  x {e}")
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
    
    # Minimal logging: remove the massive detection dump after evaluating
    try:
        os.remove(out_path)
    except:
        pass
        
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
    print(f"\nBatch size: {BATCH_SIZE}")
    print(f"Inference calls per image: ~{len(conditions)} -> ~{(len(conditions) + BATCH_SIZE - 1) // BATCH_SIZE} batched")
    print()

    defense_names = list(DEFENSES.keys())
    defense_funcs = list(DEFENSES.values())

    for fname in tqdm(files, desc="Processing images"):
        img_id = int(os.path.splitext(fname)[0])
        img_path = os.path.join(IMAGE_DIR, fname)
        pil_img = Image.open(img_path).convert("RGB")

        # ---- Prepare all image variants for this source image ----
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

        # ---- Run inference in batches of BATCH_SIZE ----
        for i in range(0, len(batch_imgs), BATCH_SIZE):
            chunk_imgs = batch_imgs[i:i + BATCH_SIZE]
            chunk_tags = batch_tags[i:i + BATCH_SIZE]
            chunk_results = run_inference_batch(chunk_imgs)

            for tag, dets in zip(chunk_tags, chunk_results):
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
print(f"{'FGSM ATTACK & DEFENSE RESULTS (VARIANT A - BEST 5)':^90}")
print("=" * 90)
print(f"\nClean Baseline:  mAP = {clean_ap:.4f},  AP50 = {clean_ap50:.4f}")
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
summary = {"clean_mAP": float(clean_ap), "clean_AP50": float(clean_ap50),
           "defense_cost": {}, "attack_results": {}}
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
ax.set_title("FGSM Attack: mAP vs Epsilon (Variant 2)", fontsize=14)
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
ax.set_title("Defense Cost on Clean Images (Variant 2)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "results_plot_v2.png"), dpi=150, bbox_inches='tight')
print(f"Plot saved to {OUTPUT_DIR}/results_plot_v2.png")

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

plt.suptitle("FGSM Attack and Defense Visual Comparison (Variant 2)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "visual_comparison_v2.png"), dpi=150, bbox_inches='tight')
print(f"Visual comparison saved to {OUTPUT_DIR}/visual_comparison_v2.png")

# ============================================================
# 12. Net Gain Analysis
# ============================================================

print("=" * 80)
print(f"{'NET GAIN ANALYSIS (VARIANT A)':^80}")
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
