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
import random
import hashlib
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

# Phase-3 locked defense set (5 solo branches -> 3 ensembles).
# DO NOT MODIFY these two names: the frozen *_v2 notebooks/scripts import
# `SOLO_DEFENSES` / `ENSEMBLES` and iterate `apply_all_defenses_gpu()` over
# them. apply_all_defenses_gpu produces exactly these 5 solo keys, so adding
# a name here (e.g. "svd") would KeyError every v2 run. The survey set lives
# in the TIER*/SURVEY_* constants below and is consumed only by run_survey_*.
SOLO_DEFENSES = ["jpeg", "median", "tvm", "gaussian", "blur_tvm"]
ENSEMBLES = {
    "ens_blur_tvm_combo":       ["jpeg", "blur_tvm", "median"],
    "ens_jpeg_median_gaussian": ["jpeg", "median", "gaussian"],
    "ens_jpeg_median_tvm":      ["jpeg", "median", "tvm"],
}

# ======================================================================
# Tiered SURVEY defense registry (additive — used only by run_survey_*).
#   Tier 1 = paper main set (the locked 5) + svd spectral filter.
#   Tier 2 = survey-completeness defenses (incl. documented negatives).
# Produced by apply_survey_defenses(); merged offline by assemble_results.
# ======================================================================
TIER1_SOLOS = SOLO_DEFENSES + ["svd"]                       # + spectral
TIER1_ENSEMBLES = {
    **ENSEMBLES,
    "ens_4way":         ["jpeg", "median", "tvm", "gaussian"],
    "ens_jpeg_tvm_svd": ["jpeg", "tvm", "svd"],
}
TIER2_SOLOS = ["nlm", "bit_depth", "random_resize", "bilateral",
               "bm3d", "dithering", "anisotropic"]
TIER2_ENSEMBLES = {
    "ens_jpeg_bilateral_tvm": ["jpeg", "bilateral", "tvm"],
    "ens_bilateral_median":   ["bilateral", "median", "tvm"],
}
SURVEY_SOLOS = TIER1_SOLOS + TIER2_SOLOS
SURVEY_ENSEMBLES = {**TIER1_ENSEMBLES, **TIER2_ENSEMBLES}
# bm3d is OPTIONAL (needs `pip install bm3d`). apply_survey_defenses DROPS it
# (with a one-time warning) when unavailable, so no mislabeled BM3D row is
# ever emitted. Survey scripts must derive their defense_names from the keys
# actually returned by apply_survey_defenses(), not from SURVEY_SOLOS blindly.
OPTIONAL_SOLOS = ["bm3d"]


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
# Tier-2 survey defense primitives.
# References:
#   bilateral  — Tomasi & Manduchi, ICCV 1998 (true joint spatial+range)
#   nlm        — Buades et al. CVPR 2005; won CAAD 2018 (Xie et al. CVPR'19)
#   bit_depth  — Xu et al. NDSS 2018 (feature squeezing)
#   random_resize — Xie et al. ICLR 2018 (input randomization)
#   dithering  — ordered (Bayer) dithering; fast stand-in for serial F-S
#   anisotropic — Perona & Malik, IEEE PAMI 1990 (edge-preserving diffusion)
#   bm3d       — Dabov et al. IEEE TIP 2007 (block-matching 3D, optional dep)
# All take/return uint8 RGB except bilateral_gpu (tensor in/out, like the
# other *_gpu primitives).
# ======================================================================
@torch.no_grad()
def bilateral_gpu(tensors: torch.Tensor,
                  sigma_color: float = 0.1,
                  sigma_space: float = 1.0,
                  radius: int | None = None) -> torch.Tensor:
    """TRUE bilateral filter (joint spatial+range Gaussian) on GPU.

    tensors: [N,C,H,W] in [0,1]. For each pixel, neighbours in a (2r+1)^2
    window are averaged with weight = spatial_gauss(distance) *
    range_gauss(|I_neighbour - I_center|), normalised per pixel. Unlike a
    plain Gaussian blur this preserves edges. (Not the mislabeled
    single-Gaussian-blend that was originally proposed.)
    """
    N, C, H, W = tensors.shape
    if radius is None:
        radius = max(1, int(np.ceil(2 * sigma_space)))
    size = 2 * radius + 1
    dev, dt = tensors.device, tensors.dtype
    ax = torch.arange(size, device=dev, dtype=dt) - radius
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    spatial = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_space ** 2))
    spatial = spatial.reshape(1, 1, size * size, 1, 1)               # [1,1,K,1,1]
    x = torch.nn.functional.pad(tensors, [radius] * 4, mode="reflect")
    patches = x.unfold(2, size, 1).unfold(3, size, 1)                # [N,C,H,W,k,k]
    patches = patches.contiguous().view(N, C, H, W, size * size)
    patches = patches.permute(0, 1, 4, 2, 3)                         # [N,C,K,H,W]
    center = tensors.unsqueeze(2)                                    # [N,C,1,H,W]
    rng = torch.exp(-((patches - center) ** 2) / (2 * sigma_color ** 2))
    w = rng * spatial
    out = (w * patches).sum(dim=2) / (w.sum(dim=2) + 1e-8)
    return out.clamp(0, 1)


def bit_depth_cpu(pil_img: Image.Image, bits: int = 4) -> Image.Image:
    """Bit-depth reduction to 2^bits levels per channel (Xu et al. NDSS 2018).

    Floor quantization, so levels are {0, f, 2f, ...} with f = 256//2^bits
    (max output 256-f, e.g. 240 for 4-bit) — a slight dimming, by design.
    """
    assert 1 <= bits <= 8, "bits must be in [1, 8]"
    arr = np.array(pil_img.convert("RGB"))
    factor = 256 // (2 ** bits)
    return Image.fromarray(((arr // factor) * factor).astype(np.uint8))


def nlm_cpu(pil_img: Image.Image,
            h_rel: float = 0.8,
            patch_size: int = 5,
            patch_distance: int = 6) -> Image.Image:
    """Non-Local Means denoising (skimage). Buades et al. 2005.

    h = h_rel * estimated noise sigma. h_rel ~ 0.8 is the usual fast_mode
    setting; the originally-proposed h=6.0*sigma over-smooths badly.
    SLOW on CPU (~0.3-3 s/image) — budget accordingly for large N.
    """
    from skimage.restoration import denoise_nl_means, estimate_sigma
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    sigma = float(estimate_sigma(arr, channel_axis=-1, average_sigmas=True))
    sigma = max(sigma, 1e-6)
    den = denoise_nl_means(arr, h=h_rel * sigma, fast_mode=True,
                           patch_size=patch_size, patch_distance=patch_distance,
                           channel_axis=-1)
    return Image.fromarray((np.clip(den, 0, 1) * 255).astype(np.uint8))


def random_resize_pad_cpu(pil_img: Image.Image,
                          scale_min: float = 0.9,
                          scale_max: float = 1.0,
                          seed: int | None = None) -> Image.Image:
    """Random resize + pad (Xie et al. ICLR 2018). Stochastic defense.

    Pass `seed` (e.g. the image id) so the SAME crop is used for the clean and
    attacked passes of one image — otherwise the recovery metric is biased and
    runs are non-reproducible. No upscaling (scale_max <= 1.0).
    """
    assert scale_max <= 1.0, "scale_max must be <= 1.0 (no upscaling supported)"
    rng = random.Random(seed)
    W, H = pil_img.size
    scale = rng.uniform(scale_min, scale_max)
    new_w, new_h = max(1, int(W * scale)), max(1, int(H * scale))
    resized = pil_img.convert("RGB").resize((new_w, new_h), Image.BICUBIC)
    out = Image.new("RGB", (W, H), (128, 128, 128))
    out.paste(resized, (rng.randint(0, W - new_w), rng.randint(0, H - new_h)))
    return out


# 4x4 Bayer threshold matrix (normalised to (0,1)) for ordered dithering.
_BAYER4 = (np.array([[0,  8,  2, 10],
                     [12, 4, 14,  6],
                     [3, 11,  1,  9],
                     [15, 7, 13,  5]], dtype=np.float32) + 0.5) / 16.0


def dithering_cpu(pil_img: Image.Image, levels: int = 16) -> Image.Image:
    """Ordered (Bayer 4x4) dithering to `levels` per channel — fully vectorised.

    A fast, DETERMINISTIC stand-in for serial Floyd-Steinberg error diffusion,
    which is O(H*W) in pure Python and infeasible at survey scale (~30-40 h for
    a 5000-image sweep). Quantises with a spatially-varying threshold, which
    disrupts adversarial-perturbation alignment like F-S dithering. Documented
    as ordered dithering — NOT Floyd-Steinberg — to keep the survey honest.
    """
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    H, W, _ = arr.shape
    step = 1.0 / levels
    thresh = np.tile(_BAYER4, (H // 4 + 1, W // 4 + 1))[:H, :W][:, :, None]
    q = np.floor(arr / step) * step
    frac = (arr - q) / step
    q = q + (frac > thresh) * step
    return Image.fromarray((np.clip(q, 0, 1) * 255).astype(np.uint8))


def anisotropic_diffusion_cpu(pil_img: Image.Image,
                              niter: int = 5,
                              kappa: float = 0.05,
                              gamma: float = 0.2) -> Image.Image:
    """Perona-Malik anisotropic diffusion — edge-preserving. PAMI 1990.

    Operates on [0,1] data, so kappa must be ~0.05 (the originally-proposed
    kappa=50 made every conductance ~1.0 => plain isotropic blur, NOT
    edge-preserving). Zero-flux (reflect) boundaries; gamma <= 0.25 for
    stability with the 4-neighbour stencil.
    """
    arr = np.array(pil_img.convert("RGB")).astype(np.float64) / 255.0
    for _ in range(niter):
        p = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
        d_n = p[:-2, 1:-1] - arr
        d_s = p[2:, 1:-1] - arr
        d_e = p[1:-1, 2:] - arr
        d_w = p[1:-1, :-2] - arr
        cn = np.exp(-(d_n / kappa) ** 2)
        cs = np.exp(-(d_s / kappa) ** 2)
        ce = np.exp(-(d_e / kappa) ** 2)
        cw = np.exp(-(d_w / kappa) ** 2)
        arr = arr + gamma * (cn * d_n + cs * d_s + ce * d_e + cw * d_w)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def bm3d_cpu(pil_img: Image.Image, sigma_psd: float = 0.1) -> Image.Image:
    """BM3D block-matching denoising (Dabov et al. TIP 2007).

    OPTIONAL: requires `pip install bm3d`. Raises RuntimeError if unavailable
    so the caller can DROP the row rather than silently emit a mislabeled
    Gaussian-blur fallback under a 'BM3D' name.
    """
    try:
        import bm3d  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError("bm3d not installed (`pip install bm3d`)") from exc
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    den = bm3d.bm3d(arr, sigma_psd=sigma_psd)
    return Image.fromarray((np.clip(den, 0, 1) * 255).astype(np.uint8))


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
# Full SURVEY defense bank (Tier 1 + Tier 2). Returns {name: PIL}.
# ======================================================================
_bm3d_warned = False


def apply_survey_defenses(pil_img: Image.Image,
                          device,
                          jpeg_quality: int = DEFAULT_JPEG_QUALITY,
                          median_kernel: int = DEFAULT_MEDIAN_KERNEL,
                          tvm_weight: float = DEFAULT_TVM_WEIGHT,
                          tvm_iters: int = DEFAULT_TVM_ITERS,
                          gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
                          svd_keep_ratio: float = DEFAULT_SVD_KEEP_RATIO,
                          seed: int | None = None) -> dict[str, Image.Image]:
    """All Tier-1 + Tier-2 survey defenses as {name: PIL image}.

    Tier-1 (GPU): the 5 locked solos (via apply_all_defenses_gpu) + svd.
    Tier-2: bilateral (GPU) + 6 CPU transforms (nlm, bit_depth, random_resize,
    dithering, anisotropic, and optionally bm3d).

    `seed` makes the stochastic random_resize deterministic per image — pass
    the image id so clean and attacked passes get the SAME crop.

    The returned dict's keys are a SUBSET of SURVEY_SOLOS: the optional 'bm3d'
    key is omitted (with a one-time warning) when bm3d is not installed, so the
    survey never reports a mislabeled BM3D result. Callers MUST build their
    defense_names from `result.keys()`, not from SURVEY_SOLOS directly.
    """
    global _bm3d_warned
    # Tier-1: 5 locked solos (one GPU trip, batched TVM) ...
    result = apply_all_defenses_gpu(
        pil_img, device,
        jpeg_quality=jpeg_quality, median_kernel=median_kernel,
        tvm_weight=tvm_weight, tvm_iters=tvm_iters,
        gaussian_sigma=gaussian_sigma,
    )
    # ... + svd + bilateral (GPU)
    src = pil_to_tensor(pil_img, device)
    result["svd"]       = tensor_to_pil(svd_gpu(src, keep_ratio=svd_keep_ratio))
    result["bilateral"] = tensor_to_pil(bilateral_gpu(src))

    # Tier-2 CPU transforms
    result["nlm"]           = nlm_cpu(pil_img)
    result["bit_depth"]     = bit_depth_cpu(pil_img)
    result["random_resize"] = random_resize_pad_cpu(pil_img, seed=seed)
    result["dithering"]     = dithering_cpu(pil_img)
    result["anisotropic"]   = anisotropic_diffusion_cpu(pil_img)
    try:
        result["bm3d"] = bm3d_cpu(pil_img)
    except RuntimeError as exc:
        if not _bm3d_warned:
            print(f"[survey] {exc} — dropping the 'bm3d' defense row.")
            _bm3d_warned = True
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
# Config signature + SAFE survey checkpoint.
#
# v1 used DetectionCheckpoint with a bare `if ckpt.has(img_id): return` over a
# single shared pickle. That silently reused stale per-image buckets when the
# attack/defense/params changed between runs -> the documented v1
# contamination (negative recovery, recovered > clean, num_images misreport).
#
# SurveyCheckpoint fixes that: every cache is stamped with a config SIGNATURE
# (attack + defense set + params + num_images). On resume it only honors a
# cache whose signature MATCHES the current run; any mismatch is discarded and
# the run starts fresh. So skipping image ids can never cross configs.
# ======================================================================
def config_signature(**fields) -> str:
    """Short stable hash of the run config. Pass everything that, if changed,
    must invalidate a resumed cache (attack, sorted defense names, eps/iters/
    patch params, num_images, image dir)."""
    blob = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class SurveyCheckpoint:
    """Config-keyed resumable per-image cache for survey runs.

    On-disk format: {"sig": <signature>, "buckets": {img_id: {tag: dets}}}.
    Set enabled=False for a pure in-memory run (no file IO, v2-style).
    """

    def __init__(self, path: str, signature: str,
                 flush_every: int = 500, enabled: bool = True):
        self.path = path
        self.signature = signature
        self.flush_every = flush_every
        self.enabled = enabled
        self.buckets: dict[int, dict[str, list]] = {}
        self._since_flush = 0
        if enabled and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    saved = pickle.load(f)
                if isinstance(saved, dict) and saved.get("sig") == signature:
                    self.buckets = saved.get("buckets", {})
                    print(f"[survey-ckpt] Resumed {len(self.buckets)} images "
                          f"(sig {signature}) from {path}")
                else:
                    print(f"[survey-ckpt] Config changed (cache sig "
                          f"{saved.get('sig') if isinstance(saved, dict) else '?'} "
                          f"!= {signature}); ignoring stale cache, starting fresh.")
            except Exception as e:  # noqa: BLE001
                print(f"[survey-ckpt] Failed to load {path}: {e}. Starting fresh.")
                self.buckets = {}
        if enabled:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def has(self, image_id: int) -> bool:
        return image_id in self.buckets

    def put(self, image_id: int, buckets: dict[str, list]):
        self.buckets[image_id] = buckets
        self._since_flush += 1
        if self.enabled and self._since_flush >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.enabled:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"sig": self.signature, "buckets": self.buckets}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)
        self._since_flush = 0
        print(f"[survey-ckpt] Flushed {len(self.buckets)} images -> {self.path}")


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
