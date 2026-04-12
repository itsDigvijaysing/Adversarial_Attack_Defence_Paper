"""
Phase 3 shared utilities: GPU-accelerated defenses, ensemble NMS merger,
checkpointing, and COCO evaluation helpers.

Used by all 6 Phase-3 notebooks (FGSM/PGD/Patch x YOLO/Florence).

Kept dependency-light (torch + numpy + PIL + pycocotools). Model-specific
code (Florence-2 / YOLO-World inference, attacks) lives in the notebooks.
"""

from __future__ import annotations
import os
import json
import pickle
import time
from io import BytesIO
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image

# ======================================================================
# Default defense parameters (can be overridden by notebook config)
# ======================================================================
DEFAULT_JPEG_QUALITY = 75
DEFAULT_MEDIAN_KERNEL = 3
DEFAULT_TVM_WEIGHT = 0.05
DEFAULT_TVM_ITERS = 200
DEFAULT_GAUSSIAN_SIGMA = 1.0
DEFAULT_SVD_KEEP_RATIO = 0.90
DEFAULT_NMS_IOU = 0.5

# Phase-3 locked defense set (5 solo branches -> 3 ensembles)
SOLO_DEFENSES = ["jpeg", "median", "tvm", "gaussian", "blur_tvm"]
ENSEMBLES = {
    "ens_blur_tvm_combo":       ["jpeg", "blur_tvm", "median"],
    "ens_jpeg_median_gaussian": ["jpeg", "median", "gaussian"],
    "ens_jpeg_median_tvm":      ["jpeg", "median", "tvm"],
}


# ======================================================================
# PIL <-> GPU tensor helpers
# ======================================================================
def pil_to_tensor(pil_img: Image.Image, device) -> torch.Tensor:
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze(0).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


# ======================================================================
# GPU-accelerated defense primitives (Chambolle TVM, separable Gaussian,
# unfold-based median, torch SVD). Match VariantZ implementations.
# ======================================================================
@torch.no_grad()
def tvm_gpu(tensors: torch.Tensor,
            weight: float = DEFAULT_TVM_WEIGHT,
            n_iter: int = DEFAULT_TVM_ITERS) -> torch.Tensor:
    N, C, H, W = tensors.shape
    x = tensors.reshape(N * C, H, W)
    p = torch.zeros(N * C, 2, H, W, device=x.device, dtype=x.dtype)
    tau = 0.25
    for _ in range(n_iter):
        d = -(p[:, 0] + p[:, 1])
        d[:, 1:, :] += p[:, 0, :-1, :]
        d[:, :, 1:] += p[:, 1, :, :-1]
        out = x + d
        g = torch.zeros_like(p)
        g[:, 0, :-1, :] = out[:, 1:, :] - out[:, :-1, :]
        g[:, 1, :, :-1] = out[:, :, 1:] - out[:, :, :-1]
        norm = torch.sqrt((g ** 2).sum(dim=1, keepdim=True))
        norm = norm * tau / weight + 1.0
        p = (p - tau * g) / norm
    d = -(p[:, 0] + p[:, 1])
    d[:, 1:, :] += p[:, 0, :-1, :]
    d[:, :, 1:] += p[:, 1, :, :-1]
    out = (x + d).clamp(0, 1)
    return out.reshape(N, C, H, W)


@torch.no_grad()
def median_gpu(tensors: torch.Tensor,
               kernel_size: int = DEFAULT_MEDIAN_KERNEL) -> torch.Tensor:
    N, C, H, W = tensors.shape
    pad = kernel_size // 2
    x = torch.nn.functional.pad(tensors, [pad] * 4, mode='reflect')
    patches = x.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    return patches.contiguous().view(N, C, H, W, -1).median(dim=-1).values


@torch.no_grad()
def svd_gpu(tensors: torch.Tensor,
            keep_ratio: float = DEFAULT_SVD_KEEP_RATIO) -> torch.Tensor:
    N, C, H, W = tensors.shape
    x = tensors.reshape(N * C, H, W) * 255.0
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    k = max(1, int(S.shape[-1] * keep_ratio))
    result = (U[:, :, :k] * S[:, :k].unsqueeze(1)) @ Vh[:, :k, :]
    return (result / 255.0).clamp(0, 1).reshape(N, C, H, W)


_gaussian_cache: dict[tuple[int, float], tuple[torch.Tensor, int]] = {}


@torch.no_grad()
def gaussian_gpu(tensors: torch.Tensor,
                 sigma: float = DEFAULT_GAUSSIAN_SIGMA) -> torch.Tensor:
    dev = tensors.device
    gpu_idx = dev.index if dev.index is not None else 0
    key = (gpu_idx, float(sigma))
    if key not in _gaussian_cache:
        radius = int(np.ceil(3 * sigma))
        size = 2 * radius + 1
        coords = torch.arange(size, device=dev, dtype=tensors.dtype) - radius
        k1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        k1d = k1d / k1d.sum()
        _gaussian_cache[key] = (k1d, radius)
    k1d, radius = _gaussian_cache[key]
    _, C, _, _ = tensors.shape
    kh = k1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    x = torch.nn.functional.pad(tensors, [radius, radius, 0, 0], mode='reflect')
    x = torch.nn.functional.conv2d(x, kh, groups=C)
    kv = k1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = torch.nn.functional.pad(x, [0, 0, radius, radius], mode='reflect')
    x = torch.nn.functional.conv2d(x, kv, groups=C)
    return x.clamp(0, 1)


def jpeg_cpu(pil_img: Image.Image,
             quality: int = DEFAULT_JPEG_QUALITY) -> Image.Image:
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ======================================================================
# Apply all 5 solo defenses (one GPU trip, batched TVM)
# ======================================================================
def apply_all_defenses_gpu(pil_img: Image.Image,
                           device,
                           jpeg_quality: int = DEFAULT_JPEG_QUALITY,
                           median_kernel: int = DEFAULT_MEDIAN_KERNEL,
                           tvm_weight: float = DEFAULT_TVM_WEIGHT,
                           tvm_iters: int = DEFAULT_TVM_ITERS,
                           gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA
                           ) -> dict[str, Image.Image]:
    """Return {solo_defense_name: PIL image} for the 5 locked solo branches.

    One GPU upload; TVM and blur_tvm share a single batched TVM call.
    """
    result: dict[str, Image.Image] = {}
    result["jpeg"] = jpeg_cpu(pil_img, jpeg_quality)

    src = pil_to_tensor(pil_img, device)                          # [1,3,H,W]
    blurred = gaussian_gpu(src, sigma=gaussian_sigma)
    tvm_in = torch.cat([src, blurred], dim=0)                     # [2,3,H,W]
    tvm_out = tvm_gpu(tvm_in, weight=tvm_weight, n_iter=tvm_iters)

    result["tvm"]      = tensor_to_pil(tvm_out[0:1])
    result["blur_tvm"] = tensor_to_pil(tvm_out[1:2])
    result["median"]   = tensor_to_pil(median_gpu(src, median_kernel))
    result["gaussian"] = tensor_to_pil(blurred)
    return result


# ======================================================================
# Ensemble merging via class-aware NMS across the union of branches
# ======================================================================
def box_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def merge_branches_nms(branch_dets_list, iou_thr: float = DEFAULT_NMS_IOU):
    """Class-aware NMS over the union of detections from several branches.

    Input: list of detection-lists (one per branch). Each detection
    {bbox:[x,y,w,h], category_id, score, image_id}.
    Output: merged list (same schema).
    """
    pooled = []
    for dets in branch_dets_list:
        pooled.extend(dets)
    if not pooled:
        return []
    buckets = defaultdict(list)
    for d in pooled:
        buckets[(d["image_id"], d["category_id"])].append(d)
    merged = []
    for _, group in buckets.items():
        group.sort(key=lambda d: d["score"], reverse=True)
        kept = []
        for d in group:
            x, y, w, h = d["bbox"]
            bx = [x, y, x + w, y + h]
            suppress = False
            for kd in kept:
                kx, ky, kw, kh = kd["bbox"]
                kbx = [kx, ky, kx + kw, ky + kh]
                if box_iou(bx, kbx) > iou_thr:
                    suppress = True
                    break
            if not suppress:
                kept.append(d)
        merged.extend(kept)
    return merged


# ======================================================================
# Checkpointing — per-image detection cache, safe to resume
# ======================================================================
class DetectionCheckpoint:
    """Pickle-backed cache: image_id -> {condition_tag: detections}.

    Flushes every `flush_every` images. Safe to resume after crash:
    loads any existing state on __init__, loop code must skip cached ids.
    """
    def __init__(self, path: str, flush_every: int = 1000):
        self.path = path
        self.flush_every = flush_every
        self.data: dict[int, dict[str, list]] = {}
        self._since_flush = 0
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.data = pickle.load(f)
                print(f"[checkpoint] Resumed {len(self.data)} images from {path}")
            except Exception as e:
                print(f"[checkpoint] Failed to load {path}: {e}. Starting fresh.")
                self.data = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def has(self, image_id: int) -> bool:
        return image_id in self.data

    def put(self, image_id: int, buckets: dict[str, list]):
        self.data[image_id] = buckets
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)
        self._since_flush = 0
        print(f"[checkpoint] Flushed {len(self.data)} images -> {self.path}")


# ======================================================================
# Assemble per-image buckets into final all_results dict, building
# ensembles offline for every attack condition.
# ======================================================================
def assemble_results(per_image: dict[int, dict[str, list]],
                     defense_names: list[str],
                     attack_tags: list[str],
                     ensembles: dict[str, list[str]] = ENSEMBLES,
                     nms_iou: float = DEFAULT_NMS_IOU):
    """Return dict[condition_tag] -> list of detections (flat across images).

    attack_tags example: ["fgsm_eps0.03"] or ["pgd_eps0.03"] or ["patch"].
    Clean + attacked ensembles built offline via NMS over solo branches.
    """
    all_results: dict[str, list] = {}
    conditions = ["clean"]
    for dname in defense_names:
        conditions.append(f"clean+{dname}")
    for atag in attack_tags:
        conditions.append(atag)
        for dname in defense_names:
            conditions.append(f"{atag}+{dname}")
    for c in conditions:
        all_results[c] = []

    for img_id, buckets in per_image.items():
        for tag, dets in buckets.items():
            if tag in all_results:
                all_results[tag].extend(dets)

    # Build ensembles for clean AND every attack
    prefixes = ["clean"] + list(attack_tags)
    for prefix in prefixes:
        for ens_name, members in ensembles.items():
            cond_tag = f"{prefix}+{ens_name}"
            merged_all = []
            for img_id, buckets in per_image.items():
                branch_dets = [buckets.get(f"{prefix}+{m}", []) for m in members]
                merged_all.extend(merge_branches_nms(branch_dets, iou_thr=nms_iou))
            all_results[cond_tag] = merged_all
    return all_results


# ======================================================================
# COCO evaluation (mAP@[0.5:0.95])
# ======================================================================
def coco_eval_tag(coco_gt, results_list, image_ids, out_path: str):
    if not results_list:
        return np.zeros(12)
    with open(out_path, "w") as f:
        json.dump(results_list, f)
    from pycocotools.cocoeval import COCOeval
    coco_dt = coco_gt.loadRes(out_path)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = image_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats


def evaluate_all_conditions(all_results, coco_gt, image_ids, output_dir: str):
    """Run COCO eval on every condition tag. Returns {tag: stats (12,)}."""
    eval_stats: dict[str, np.ndarray] = {}
    for tag, results_list in all_results.items():
        print(f"\n{'=' * 70}\n[COCO] {tag} ({len(results_list)} detections)\n{'=' * 70}")
        out_path = os.path.join(output_dir, f"{tag.replace('+','_').replace('/','_')}.json")
        eval_stats[tag] = coco_eval_tag(coco_gt, results_list, image_ids, out_path)
    return eval_stats


# ======================================================================
# GPU isolation helper (picks the GPU with most free memory from a pool)
# ======================================================================
def pick_best_gpu(pool: list[int] | None = None) -> str:
    """Return the index (as string) of the freest GPU in `pool`.

    Call BEFORE `import torch`. Set `os.environ['CUDA_VISIBLE_DEVICES']`
    with the returned string.
    """
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
        if pool is not None:
            gpu_free = [g for g in gpu_free if g[0] in pool]
        gpu_free.sort(key=lambda x: x[1], reverse=True)
        if not gpu_free:
            return "0"
        return str(gpu_free[0][0])
    except Exception:
        return "0"


# ======================================================================
# 35x35 center-placed adversarial patch — single consistent spec
# ======================================================================
PATCH_SIZE = 35
PATCH_OPT_ITERS = 100
PATCH_LR = 0.02


def place_patch(img_tensor: torch.Tensor,
                patch: torch.Tensor,
                top: int, left: int) -> torch.Tensor:
    """Composite a patch onto an image tensor in-place style (returns new).

    img_tensor: [1,3,H,W] in [0,1]. patch: [3,ph,pw] in [0,1].
    """
    out = img_tensor.clone()
    _, _, H, W = out.shape
    ph, pw = patch.shape[1], patch.shape[2]
    out[:, :, top:top + ph, left:left + pw] = patch
    return out


def center_patch_coords(H: int, W: int, patch_size: int = PATCH_SIZE):
    top = (H - patch_size) // 2
    left = (W - patch_size) // 2
    return top, left


# ======================================================================
# Print banners
# ======================================================================
def print_banner(title: str, char: str = "=", width: int = 70):
    print(char * width)
    print(f"{title:^{width}}")
    print(char * width)
