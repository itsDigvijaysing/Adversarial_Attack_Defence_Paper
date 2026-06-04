#!/usr/bin/env python3
"""
Survey-scale Florence-2 OCR robustness sweep (Ankush) — FGSM + PGD + Patch in
ONE process, model loaded ONCE, over the full Tier-1 + Tier-2 survey defense
bank from phase3_common.apply_survey_defenses().

------------------------------------------------------------------------------
METRIC — read this carefully, it is NOT OCR accuracy.
------------------------------------------------------------------------------
The reported number is `text_similarity` = char-level difflib SequenceMatcher
ratio of (the defended-image OCR text) vs (the model's OWN clean-image OCR
output, `clean_ref`). It is a SELF-CONSISTENCY / PREDICTION-STABILITY recovery
score: "how close is the OCR string after attack+defense to what the SAME model
produced on the clean image". It is NOT OCR accuracy (no ground-truth text is
used) and it is NOT a word-recovery ratio. `clean_baseline` is the deterministic
clean-vs-clean self-consistency (~1.0). This exact description is also stamped
into every summary_{attack}.json under the "metric" field.

------------------------------------------------------------------------------
Defenses (all via pc.apply_survey_defenses, Tier-1 + Tier-2):
  jpeg          — Dziugaite et al. 2016 (JPEG compression as a defense)
  median        — Xu et al. NDSS 2018 (feature squeezing: median smoothing)
  gaussian      — Xu et al. NDSS 2018 (local smoothing)
  tvm           — Guo et al. ICLR 2018 (total variation minimization)
  blur_tvm      — gaussian -> TVM composite (this work)
  svd           — low-rank SVD spectral filter (this work)
  bilateral     — Tomasi & Manduchi, ICCV 1998 (edge-preserving range filter)
  nlm           — Buades et al. CVPR 2005 (non-local means)
  bit_depth     — Xu et al. NDSS 2018 (feature squeezing: bit-depth reduction)
  random_resize — Xie et al. ICLR 2018 (input randomization)
  dithering     — ordered (Bayer 4x4) dithering; FAST DETERMINISTIC stand-in
                  for serial Floyd-Steinberg error diffusion
  anisotropic   — Perona & Malik, IEEE PAMI 1990 (anisotropic diffusion)
  bm3d          — Dabov et al. IEEE TIP 2007 (OPTIONAL; the 'bm3d' key is
                  omitted by apply_survey_defenses when bm3d is not installed,
                  so this script derives defense_names from the returned keys)
  sign_approx   — only with --novel. APPROXIMATION of SIGN (arXiv:2605.27927,
                  2026 preprint): top-0.5% highest-|gradient| pixels of the
                  attacked image are replaced by their local 3x3 median. This
                  is an approximation (gradient saliency proxy, not the full
                  SIGN procedure) and is labeled 'sign_approx' accordingly.

Attacks:
  FGSM  — Goodfellow et al. ICLR 2015      (eps=0.03, normalized pixel space)
  PGD   — Madry et al. ICLR 2018           (eps=0.03, 10 iters, alpha=eps/4,
                                            random start, L-inf, normalized)
  Patch — Brown et al. 2017                (35x35 center patch, 100 Adam steps,
                                            lr=0.02)

Novel defenses citation block (only sign_approx is implemented here):
  SIGN      — arXiv:2605.27927 (2026, preprint)  [APPROXIMATION via sign_approx]
  PAD       — CVPR 2024, arXiv:2404.16452 (SAM-based patch defense) [not run here]
  XAIAD-YOLO— Future Generation Computer Systems 2026,
              DOI 10.1016/j.future.2025.108356                       [not run here]

------------------------------------------------------------------------------
The OCR self-consistency metric, normalize_text, text_similarity (difflib
SequenceMatcher), decode_ocr, sanitize_generated_text, get_ocr_text (prompt
'<OCR>'), consensus_vote_text, and the per-channel norm_min/norm_max FGSM clamp
are reused VERBATIM from FGSM_Florence2_OCR_Robust.py / PGD_/Patch_ siblings.
DO NOT execute this file here (no GPU / model / data); syntax-check only.
------------------------------------------------------------------------------
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List

# ======================================================================
# 1. GPU isolation — MUST happen BEFORE importing torch / phase3_common.
#    Scan sys.argv for --gpu early. If gpu >= 0 pin it; if gpu < 0 (or
#    unset default) pick the freest GPU via an inline nvidia-smi probe,
#    mirroring the *_v2 scripts.
# ======================================================================
def _early_gpu_arg(argv) -> int:
    gpu = -1
    for i, tok in enumerate(argv):
        if tok == "--gpu" and i + 1 < len(argv):
            try:
                gpu = int(argv[i + 1])
            except ValueError:
                gpu = -1
        elif tok.startswith("--gpu="):
            try:
                gpu = int(tok.split("=", 1)[1])
            except ValueError:
                gpu = -1
    return gpu


def _pick_freest_gpu() -> str:
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
        if gpu_free:
            return str(gpu_free[0][0])
    except Exception as exc:  # noqa: BLE001
        print(f"[GPU] nvidia-smi probe failed ({exc}); using default GPU 0.")
    return "0"


_GPU = _early_gpu_arg(sys.argv)
if _GPU >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_GPU)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pick_freest_gpu()
print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# Now safe to import torch + phase3_common (they read CUDA_VISIBLE_DEVICES).
import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
from PIL import Image                                           # noqa: E402
from transformers import AutoModelForCausalLM, AutoProcessor    # noqa: E402

import phase3_common as pc                                      # noqa: E402


# ======================================================================
# 2. OCR helpers — reused VERBATIM from the *_OCR_Robust.py base scripts.
# ======================================================================
def print_banner(title: str, char: str = "=", width: int = 100) -> None:
    print(char * width)
    print(f"{title:^{width}}")
    print(char * width)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def decode_ocr(parsed_or_text) -> str:
    if isinstance(parsed_or_text, dict):
        val = parsed_or_text.get("<OCR>", "")
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            return " ".join(str(x) for x in val)
        if isinstance(val, dict):
            if "text" in val:
                return str(val["text"])
            return json.dumps(val, ensure_ascii=True)
    return str(parsed_or_text)


def sanitize_generated_text(raw: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", cleaned).strip()


def consensus_vote_text(texts: List[str]) -> str:
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    norm = [normalize_text(t) for t in texts]
    counts = Counter(norm)
    top_norm, top_count = counts.most_common(1)[0]
    tied = [k for k, v in counts.items() if v == top_count]

    if len(tied) == 1 and top_count >= 2:
        for t in texts:
            if normalize_text(t) == top_norm:
                return t

    pairwise = []
    for i, ti in enumerate(texts):
        sim_sum = 0.0
        for j, tj in enumerate(texts):
            if i == j:
                continue
            sim_sum += text_similarity(ti, tj)
        pairwise.append(sim_sum)

    best_idx = max(range(len(texts)),
                   key=lambda i: (pairwise[i], len(norm[i]), len(texts[i])))
    return texts[best_idx]


def get_ocr_text(
    model,
    processor,
    pil_img: Image.Image,
    device,
    torch_dtype,
    max_new_tokens: int,
    num_beams: int,
    repetition_penalty: float,
    length_penalty: float,
) -> str:
    img_w, img_h = pil_img.size
    with torch.inference_mode():
        inputs = processor(text="<OCR>", images=pil_img, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype)
        gen_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
        raw = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        try:
            parsed = processor.post_process_generation(
                raw, task="<OCR>", image_size=(img_w, img_h))
            return decode_ocr(parsed)
        except Exception:
            return sanitize_generated_text(raw)


# ======================================================================
# 3. Attacks — FGSM / PGD / Patch in normalized pixel space.
#    FGSM reused verbatim; PGD & Patch extend the same structure (the
#    per-channel norm_min/norm_max clamp is identical across all three).
# ======================================================================
def fgsm_attack_ocr(
    model, processor, pil_img, eps, device, torch_dtype,
    max_new_tokens, num_beams, repetition_penalty, length_penalty,
    img_mean, img_std, norm_min, norm_max,
) -> Image.Image:
    orig_size = pil_img.size
    inputs = processor(text="<OCR>", images=pil_img, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype)

    with torch.no_grad():
        target_ids = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            do_sample=False, repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
    target_ids = target_ids[:1, :].contiguous()
    if target_ids.size(1) > max_new_tokens:
        target_ids = target_ids[:, :max_new_tokens].contiguous()

    model.zero_grad(set_to_none=True)
    pixel_values_adv = pixel_values.clone().detach().requires_grad_(True)
    outputs = model(input_ids=input_ids, pixel_values=pixel_values_adv,
                    labels=target_ids)
    outputs.loss.backward()

    grad = pixel_values_adv.grad
    if grad is None:
        raise RuntimeError("FGSM gradient is None.")

    adv_pixel_values = pixel_values.detach() + eps * grad.sign()
    adv_pixel_values = torch.max(torch.min(adv_pixel_values, norm_max), norm_min)

    adv_denorm = torch.clamp(adv_pixel_values * img_std + img_mean, 0.0, 1.0)
    adv_np = (adv_denorm.squeeze(0).permute(1, 2, 0).detach()
              .cpu().float().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    if adv_pil.size != orig_size:
        adv_pil = adv_pil.resize(orig_size, Image.BICUBIC)
    return adv_pil


def pgd_attack_ocr(
    model, processor, pil_img, eps, alpha, steps, random_start,
    device, torch_dtype,
    max_new_tokens, num_beams, repetition_penalty, length_penalty,
    img_mean, img_std, norm_min, norm_max,
) -> Image.Image:
    orig_size = pil_img.size
    inputs = processor(text="<OCR>", images=pil_img, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype).detach()

    with torch.no_grad():
        target_ids = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            do_sample=False, repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
    target_ids = target_ids[:1, :].contiguous()
    if target_ids.size(1) > max_new_tokens:
        target_ids = target_ids[:, :max_new_tokens].contiguous()

    x_orig = pixel_values.clone().detach()
    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    else:
        x_adv = x_orig.clone()
    x_adv = torch.max(torch.min(x_adv, norm_max), norm_min)

    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        x_adv = x_adv.detach().requires_grad_(True)
        outputs = model(input_ids=input_ids, pixel_values=x_adv,
                        labels=target_ids)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError("PGD loss became non-finite.")
        loss.backward()
        grad = x_adv.grad
        if grad is None:
            raise RuntimeError("PGD gradient is None.")
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = x_orig + delta
        x_adv = torch.max(torch.min(x_adv, norm_max), norm_min)

    adv_pixel_values = x_adv.detach()
    adv_denorm = torch.clamp(adv_pixel_values * img_std + img_mean, 0.0, 1.0)
    adv_np = (adv_denorm.squeeze(0).permute(1, 2, 0).detach()
              .cpu().float().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    if adv_pil.size != orig_size:
        adv_pil = adv_pil.resize(orig_size, Image.BICUBIC)
    return adv_pil


def patch_attack_ocr(
    model, processor, pil_img, patch_size, patch_iters, patch_lr,
    random_location, device, torch_dtype,
    max_new_tokens, num_beams, repetition_penalty, length_penalty,
    img_mean, img_std, norm_min, norm_max,
) -> Image.Image:
    orig_size = pil_img.size
    inputs = processor(text="<OCR>", images=pil_img, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype).detach()

    with torch.no_grad():
        target_ids = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            do_sample=False, repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
    target_ids = target_ids[:1, :].contiguous()
    if target_ids.size(1) > max_new_tokens:
        target_ids = target_ids[:, :max_new_tokens].contiguous()

    x_orig = pixel_values.to(dtype=torch.float32)
    img_mean32 = img_mean.to(dtype=torch.float32)
    img_std32 = img_std.to(dtype=torch.float32)
    norm_min32 = norm_min.to(dtype=torch.float32)
    norm_max32 = norm_max.to(dtype=torch.float32)

    _, _, h, w = x_orig.shape
    ps = max(1, min(int(patch_size), h, w))
    if random_location:
        top = int(np.random.randint(0, h - ps + 1))
        left = int(np.random.randint(0, w - ps + 1))
    else:
        top = (h - ps) // 2
        left = (w - ps) // 2

    patch_param = torch.empty((1, 3, ps, ps), device=device, dtype=torch.float32)
    patch_param.uniform_(0.0, 1.0)
    patch_param.requires_grad_(True)
    optimizer = torch.optim.Adam([patch_param], lr=patch_lr)

    for _ in range(patch_iters):
        optimizer.zero_grad(set_to_none=True)
        patch_img = patch_param.clamp(0.0, 1.0)
        patch_norm = (patch_img - img_mean32) / img_std32
        x_adv32 = x_orig.clone()
        x_adv32[:, :, top:top + ps, left:left + ps] = patch_norm
        x_adv32 = torch.max(torch.min(x_adv32, norm_max32), norm_min32)
        x_adv = x_adv32.to(dtype=torch_dtype)
        outputs = model(input_ids=input_ids, pixel_values=x_adv,
                        labels=target_ids)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError("Patch attack loss became non-finite.")
        (-loss).backward()  # maximize task loss to degrade OCR
        if patch_param.grad is None:
            raise RuntimeError("Patch attack gradient is None.")
        optimizer.step()

    with torch.no_grad():
        patch_img = patch_param.clamp(0.0, 1.0)
        patch_norm = (patch_img - img_mean32) / img_std32
        x_adv32 = x_orig.clone()
        x_adv32[:, :, top:top + ps, left:left + ps] = patch_norm
        x_adv32 = torch.max(torch.min(x_adv32, norm_max32), norm_min32)

    adv_denorm = torch.clamp(x_adv32 * img_std32 + img_mean32, 0.0, 1.0)
    adv_np = (adv_denorm.squeeze(0).permute(1, 2, 0).detach()
              .cpu().float().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    if adv_pil.size != orig_size:
        adv_pil = adv_pil.resize(orig_size, Image.BICUBIC)
    return adv_pil


# ======================================================================
# 4. Novel defense: sign_approx — APPROXIMATION of SIGN (arXiv:2605.27927).
#    Top-0.5% highest-|gradient| pixels of the attacked image are replaced
#    by their local 3x3 median. Pure-numpy, no extra deps; the try/except in
#    the caller still guards it so a missing dep can never crash the run.
# ======================================================================
def sign_approx_defense(adv_pil: Image.Image, top_frac: float = 0.005) -> Image.Image:
    arr = np.array(adv_pil.convert("RGB")).astype(np.float32)
    H, W, _ = arr.shape
    gray = arr.mean(axis=2)
    gy, gx = np.gradient(gray)
    gmag = np.sqrt(gx * gx + gy * gy)

    # 3x3 reflect-padded median for every pixel/channel.
    pad = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    stack = np.stack(
        [pad[dy:dy + H, dx:dx + W, :]
         for dy in range(3) for dx in range(3)],
        axis=0,
    )
    med = np.median(stack, axis=0)

    if gmag.size > 0:
        thresh = np.quantile(gmag, 1.0 - top_frac)
        mask = (gmag >= thresh)[:, :, None]
        out = np.where(mask, med, arr)
    else:
        out = arr
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ======================================================================
# 5. SAFE resumable per-attack state — one state_{attack}.json, stamped
#    with a config signature. On load, if the stored sig != current sig,
#    START FRESH (do not reuse). Atomic tmp + os.replace save.
# ======================================================================
def make_default_state(sig: str, defense_names: List[str]) -> Dict:
    return {
        "sig": sig,
        "processed": [],
        "failed": [],
        "clean_baseline_sum": 0.0,
        "attacked_sim_sum": 0.0,
        "defenses": {
            name: {"sim_sum": 0.0, "recovery_sum": 0.0}
            for name in defense_names
        },
    }


def load_state(state_path: str, sig: str, defense_names: List[str]) -> Dict:
    if not os.path.isfile(state_path):
        return make_default_state(sig, defense_names)
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"[state] Failed to read {state_path}: {exc}. Starting fresh.")
        return make_default_state(sig, defense_names)

    if state.get("sig") != sig:
        print(f"[state] Config changed (stored sig {state.get('sig')} != "
              f"{sig}); ignoring stale state, starting fresh.")
        return make_default_state(sig, defense_names)

    for key in ["processed", "failed", "clean_baseline_sum",
                "attacked_sim_sum", "defenses"]:
        if key not in state:
            return make_default_state(sig, defense_names)
    # Backfill any newly-present defense names (e.g. bm3d appearing).
    for name in defense_names:
        state["defenses"].setdefault(name, {"sim_sum": 0.0, "recovery_sum": 0.0})
    return state


def save_state(state_path: str, state: Dict) -> None:
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, state_path)


# ======================================================================
# 6. Misc helpers
# ======================================================================
def resolve_image_dir(primary_dir: str) -> str:
    candidates = [
        primary_dir,
        "Dataset/val2017", "./Dataset/val2017",
        "dataset/val2017", "./dataset/val2017",
        "val2017", "./val2017",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise RuntimeError("Could not find image directory. Tried: "
                       + ", ".join(candidates))


def verdict_from_recovery(recovery: float) -> str:
    if recovery > 0.005:
        return "RECOVERS"
    if recovery > 0.0:
        return "marginal"
    if recovery > -0.005:
        return "neutral"
    return "HURTS"


def image_id_from_name(fname: str) -> int:
    """COCO-style 000000XXXXXX.jpg -> int(XXXXXX); fallback to a stable hash.
    Used only as the deterministic seed for the stochastic random_resize."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    digits = re.sub(r"\D", "", stem)
    if digits:
        try:
            return int(digits)
        except ValueError:
            pass
    return abs(hash(stem)) % (2 ** 31)


# ======================================================================
# 7. Per-attack run. Model/processor/defense bank are passed in (loaded
#    ONCE in run_all). Returns the final state dict for the attack.
# ======================================================================
def run_one_attack(attack, args, model, processor, device, torch_dtype,
                   img_mean, img_std, norm_min, norm_max,
                   files, image_dir, ensembles_full, use_novel):
    out_dir = args.output_dir
    state_path = os.path.join(out_dir, f"state_{attack}.json")
    summary_path = os.path.join(out_dir, f"summary_{attack}.json")
    failed_path = os.path.join(out_dir, f"failed_{attack}.json")

    gen_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
    }

    pgd_alpha = args.eps / 4.0  # alpha = eps/4 per spec

    # ---- Probe the defense bank ONCE on the first image to learn the exact
    #      keys actually produced (bm3d may be absent). Derive defense_names
    #      from these keys; filter ensembles to those whose members all exist.
    probe_img = Image.open(os.path.join(image_dir, files[0])).convert("RGB")
    probe_defs = pc.apply_survey_defenses(
        probe_img, device, seed=image_id_from_name(files[0]))
    branch_names = list(probe_defs.keys())
    if use_novel:
        # sign_approx is an extra branch appended after survey defenses.
        branch_names = branch_names + ["sign_approx"]
    ensembles = {
        name: members for name, members in ensembles_full.items()
        if all(m in branch_names for m in members)
    }
    defense_names = branch_names + list(ensembles.keys())

    # ---- SAFE config signature (sorted defense names so order-independent).
    sig = pc.config_signature(
        model=args.model_name,
        attack=attack,
        defenses=sorted(defense_names),
        eps=args.eps,
        pgd_iters=args.pgd_iters,
        pgd_alpha=pgd_alpha,
        patch_size=args.patch_size,
        patch_iters=args.patch_iters,
        patch_lr=args.patch_lr,
        num_images=args.num_images,
        image_dir=image_dir,
    )

    if args.no_checkpoint and os.path.isfile(state_path):
        # --no-checkpoint => pure in-memory; ignore any prior state file.
        os.remove(state_path)
    state = (make_default_state(sig, defense_names)
             if args.no_checkpoint
             else load_state(state_path, sig, defense_names))
    state["sig"] = sig
    for name in defense_names:
        state["defenses"].setdefault(name, {"sim_sum": 0.0, "recovery_sum": 0.0})

    processed_set = set(state["processed"])
    failed_set = {m.split(":", 1)[0] for m in state["failed"]}
    pending = [f for f in files if f not in processed_set and f not in failed_set]

    print_banner(f"SURVEY OCR ATTACK = {attack}  (tier={args.tier}, "
                 f"novel={use_novel})", width=100)
    print(f"  defenses ({len(defense_names)}): {', '.join(defense_names)}")
    print(f"  ensembles kept: {', '.join(ensembles.keys()) or '(none)'}")
    print(f"  sig={sig} | already done={len(processed_set)} | "
          f"failed={len(failed_set)} | pending={len(pending)}")
    if attack == "fgsm":
        print(f"  FGSM eps={args.eps}")
    elif attack == "pgd":
        print(f"  PGD eps={args.eps} iters={args.pgd_iters} alpha={pgd_alpha:.5f} "
              f"random_start=True")
    elif attack == "patch":
        print(f"  Patch {args.patch_size}x{args.patch_size} center, "
              f"iters={args.patch_iters} lr={args.patch_lr}")

    start = time.time()
    best_running = float("-inf")
    for i, fname in enumerate(pending, 1):
        path = os.path.join(image_dir, fname)
        img_id = image_id_from_name(fname)
        try:
            clean_img = Image.open(path).convert("RGB")

            # clean_ref = the model's OWN clean OCR output (the metric anchor).
            clean_ref = get_ocr_text(model, processor, clean_img, device,
                                     torch_dtype, **gen_cfg)

            if args.skip_clean_consistency:
                clean_baseline = 1.0
            else:
                clean_again = get_ocr_text(model, processor, clean_img, device,
                                           torch_dtype, **gen_cfg)
                clean_baseline = text_similarity(clean_ref, clean_again)

            # ---- attack ----
            if attack == "fgsm":
                adv_img = fgsm_attack_ocr(
                    model, processor, clean_img, eps=args.eps, device=device,
                    torch_dtype=torch_dtype, img_mean=img_mean, img_std=img_std,
                    norm_min=norm_min, norm_max=norm_max, **gen_cfg)
            elif attack == "pgd":
                adv_img = pgd_attack_ocr(
                    model, processor, clean_img, eps=args.eps, alpha=pgd_alpha,
                    steps=args.pgd_iters, random_start=True, device=device,
                    torch_dtype=torch_dtype, img_mean=img_mean, img_std=img_std,
                    norm_min=norm_min, norm_max=norm_max, **gen_cfg)
            elif attack == "patch":
                adv_img = patch_attack_ocr(
                    model, processor, clean_img, patch_size=args.patch_size,
                    patch_iters=args.patch_iters, patch_lr=args.patch_lr,
                    random_location=False, device=device, torch_dtype=torch_dtype,
                    img_mean=img_mean, img_std=img_std, norm_min=norm_min,
                    norm_max=norm_max, **gen_cfg)
            else:
                raise RuntimeError(f"Unknown attack '{attack}'")

            attacked_text = get_ocr_text(model, processor, adv_img, device,
                                         torch_dtype, **gen_cfg)
            attacked_sim = text_similarity(clean_ref, attacked_text)

            # ---- all survey-defense branches (seed=img_id => deterministic) ----
            defended = pc.apply_survey_defenses(adv_img, device, seed=img_id)
            if use_novel:
                defended["sign_approx"] = sign_approx_defense(adv_img)

            branch_text = {}
            defense_sim = {}
            for name in branch_names:
                d_img = defended[name]
                d_txt = get_ocr_text(model, processor, d_img, device,
                                     torch_dtype, **gen_cfg)
                branch_text[name] = d_txt
                defense_sim[name] = text_similarity(clean_ref, d_txt)

            for ens_name, members in ensembles.items():
                voted = consensus_vote_text([branch_text[m] for m in members])
                defense_sim[ens_name] = text_similarity(clean_ref, voted)

            state["clean_baseline_sum"] += clean_baseline
            state["attacked_sim_sum"] += attacked_sim
            for name, sim in defense_sim.items():
                state["defenses"][name]["sim_sum"] += sim
                state["defenses"][name]["recovery_sum"] += (sim - attacked_sim)

            state["processed"].append(fname)
            best_running = max(best_running, max(defense_sim.values(), default=0.0))

        except Exception as exc:  # noqa: BLE001 — one bad image never kills the run
            state["failed"].append(f"{fname}: {exc}")

        # ---- flush cadence (skipped entirely when --no-checkpoint) ----
        if not args.no_checkpoint and (
                i % args.checkpoint_every == 0 or i == len(pending)):
            save_state(state_path, state)

        # ---- MINIMAL logging: one line every 100 images (and at the end) ----
        if i % 100 == 0 or i == len(pending):
            done = len(state["processed"])
            atk_avg = state["attacked_sim_sum"] / done if done else 0.0
            best = best_running if best_running > float("-inf") else 0.0
            elapsed_min = (time.time() - start) / 60.0
            print(f"[{attack}] {i}/{len(pending)} | attacked={atk_avg:.4f} | "
                  f"best={best:.4f} | {elapsed_min:.1f}m")

        if (torch.cuda.is_available() and args.empty_cache_every > 0
                and i % args.empty_cache_every == 0):
            torch.cuda.empty_cache()

    if not args.no_checkpoint:
        save_state(state_path, state)

    # ==================================================================
    # HARD ASSERT (documented exactly):
    #   processed + failed must cover EVERY requested image exactly once,
    #   and if there were NO failures, processed MUST equal num_images.
    # ==================================================================
    n_proc = len(state["processed"])
    n_fail = len(state["failed"])
    requested = len(files)  # == num_images (files already truncated)
    proc_names = set(state["processed"])
    fail_names = {m.split(":", 1)[0] for m in state["failed"]}
    covered = proc_names | fail_names
    assert covered == set(files), (
        f"[{attack}] coverage mismatch: processed+failed cover {len(covered)} "
        f"images but {requested} were requested "
        f"(missing={len(set(files) - covered)}, extra={len(covered - set(files))})."
    )
    if n_fail == 0:
        assert n_proc == args.num_images, (
            f"[{attack}] processed={n_proc} != num_images={args.num_images} "
            f"with zero failures.")
    else:
        print(f"[{attack}] NOTE: {n_fail} image(s) failed; "
              f"processed({n_proc}) + failed({n_fail}) = "
              f"{n_proc + n_fail} == requested({requested}).")

    # ==================================================================
    # Ranked defense table + summary_{attack}.json
    # ==================================================================
    n = max(1, n_proc)
    clean_avg = state["clean_baseline_sum"] / n
    attacked_avg = state["attacked_sim_sum"] / n
    attack_drop = clean_avg - attacked_avg

    rows = []
    for name in defense_names:
        sim_avg = state["defenses"][name]["sim_sum"] / n
        rec_avg = state["defenses"][name]["recovery_sum"] / n
        rec_pct = (100.0 * rec_avg / attack_drop) if attack_drop > 1e-12 else 0.0
        kind = "ENSEMBLE" if name in ensembles else "solo"
        rows.append({
            "name": name, "kind": kind,
            "attacked": attacked_avg, "recovered": sim_avg,
            "delta": rec_avg, "recovery_pct": rec_pct,
            "verdict": verdict_from_recovery(rec_avg),
        })
    rows.sort(key=lambda x: x["delta"], reverse=True)

    metric_desc = (
        "text_similarity = char-level difflib SequenceMatcher ratio of the "
        "defended-image OCR text vs the model's OWN clean-image OCR output "
        "(clean_ref). This is SELF-CONSISTENCY / prediction-stability recovery, "
        "NOT OCR accuracy and NOT a word-recovery ratio. clean_baseline is the "
        "deterministic clean-vs-clean self-consistency (~1.0)."
    )
    summary = {
        "model": args.model_name,
        "task": "OCR",
        "attack": attack,
        "tier": args.tier,
        "novel": use_novel,
        "metric": metric_desc,
        "epsilon": args.eps,
        "pgd_iters": args.pgd_iters,
        "pgd_alpha": pgd_alpha,
        "patch_size": args.patch_size,
        "patch_iters": args.patch_iters,
        "patch_lr": args.patch_lr,
        "images": n_proc,
        "num_images_requested": args.num_images,
        "failed_images": n_fail,
        "clean_baseline": clean_avg,
        "attacked": attacked_avg,
        "attack_damage": attack_drop,
        "defense_names": defense_names,
        "ensembles": ensembles,
        "ranked_defenses": rows,
        "config": {
            "image_dir": image_dir,
            "jpeg_quality": pc.DEFAULT_JPEG_QUALITY,
            "median_kernel": pc.DEFAULT_MEDIAN_KERNEL,
            "tvm_weight": pc.DEFAULT_TVM_WEIGHT,
            "gaussian_sigma": pc.DEFAULT_GAUSSIAN_SIGMA,
            "svd_keep_ratio": pc.DEFAULT_SVD_KEEP_RATIO,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "repetition_penalty": args.repetition_penalty,
            "length_penalty": args.length_penalty,
            "config_signature": sig,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(state["failed"], f, indent=2)

    print()
    print_banner(f"ATTACKED vs RECOVERED - {attack} (FLORENCE-2 OCR SURVEY)",
                 width=100)
    print(f"  Clean baseline   : {clean_avg:.4f}")
    print(f"  Attacked         : {attacked_avg:.4f}")
    print(f"  Attack damage    : {attack_drop:+.4f}\n")
    print("-" * 100)
    print(f"  {'Rank':<5} {'Defense':<28} {'Kind':<10} {'Attacked':>9} "
          f"{'Recovered':>10} {'DeltaSim':>9} {'Rec%':>7} {'Verdict':>10}")
    print("-" * 100)
    for rank, row in enumerate(rows, 1):
        print(f"  {rank:<5} {row['name']:<28} {row['kind']:<10} "
              f"{row['attacked']:>9.4f} {row['recovered']:>10.4f} "
              f"{row['delta']:>+9.4f} {row['recovery_pct']:>+6.1f}% "
              f"{row['verdict']:>10}")
    print("-" * 100)
    if rows:
        best, worst = rows[0], rows[-1]
        print(f"\n  >> BEST  : {best['name']} ({best['kind']})  "
              f"sim {attacked_avg:.4f} -> {best['recovered']:.4f}  "
              f"({best['delta']:+.4f}, {best['recovery_pct']:+.1f}%)")
        print(f"  >> WORST : {worst['name']} ({worst['kind']})  "
              f"sim {attacked_avg:.4f} -> {worst['recovered']:.4f}  "
              f"({worst['delta']:+.4f})")
    print(f"  Saved: {summary_path}\n")
    return state


# ======================================================================
# 8. run_all — load model ONCE, run every requested attack sequentially.
# ======================================================================
def run_all(args) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    image_dir = resolve_image_dir(args.image_dir)
    files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    )
    files = files[:args.num_images]
    if not files:
        raise RuntimeError(f"No images found in {image_dir}")
    if len(files) < args.num_images:
        print(f"[WARN] requested num_images={args.num_images} but only "
              f"{len(files)} images present; treating {len(files)} as the "
              f"requested set for the hard assert.")
        args.num_images = len(files)

    assert torch.cuda.is_available(), "CUDA required for the survey OCR run."
    device = torch.device("cuda:0")
    torch_dtype = torch.float16

    # ---- Tier selection: survey -> SURVEY ensembles; tier1 -> TIER1.
    if args.tier == "survey":
        ensembles_full = pc.SURVEY_ENSEMBLES
    else:  # tier1
        ensembles_full = pc.TIER1_ENSEMBLES
    # NOTE: apply_survey_defenses ALWAYS returns the full Tier1+Tier2 branch
    # bank; --tier only governs which ensemble set we attempt. The ensemble
    # filter below (in run_one_attack) drops any ensemble referencing a branch
    # that is absent (e.g. bm3d), so tier1 + survey both stay self-consistent.

    print_banner("FLORENCE-2 OCR ROBUSTNESS SURVEY (Ankush)", width=100)
    print(f"  image_dir   : {image_dir}")
    print(f"  num_images  : {args.num_images}")
    print(f"  attacks     : {', '.join(args.attacks)}")
    print(f"  tier        : {args.tier}")
    print(f"  novel       : {args.novel}")
    print(f"  device      : {device} | dtype: {torch_dtype}")
    print(f"  model       : {args.model_name} @ {args.model_revision}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  checkpoint  : {'OFF' if args.no_checkpoint else 'ON'} "
          f"(flush every {args.checkpoint_every})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, revision=args.model_revision,
        torch_dtype=torch_dtype, trust_remote_code=True,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_name, revision=args.model_revision, trust_remote_code=True)

    img_mean = torch.tensor(processor.image_processor.image_mean,
                            device=device, dtype=torch_dtype).view(1, 3, 1, 1)
    img_std = torch.tensor(processor.image_processor.image_std,
                           device=device, dtype=torch_dtype).view(1, 3, 1, 1)
    norm_min = (0.0 - img_mean) / img_std
    norm_max = (1.0 - img_mean) / img_std

    # ---- --novel: guard the SIGN approximation behind a try/except probe so a
    #      broken dependency (none here — pure numpy) is warned-and-skipped,
    #      never crashing the run.
    use_novel = False
    if args.novel:
        try:
            _t = sign_approx_defense(Image.new("RGB", (16, 16), (128, 128, 128)))
            assert _t.size == (16, 16)
            use_novel = True
            print("[novel] sign_approx (approximation of SIGN arXiv:2605.27927) "
                  "enabled.")
        except Exception as exc:  # noqa: BLE001
            print(f"[novel] WARNING: sign_approx unavailable ({exc}); "
                  f"skipping the novel defense.")
            use_novel = False

    for attack in args.attacks:
        run_one_attack(
            attack, args, model, processor, device, torch_dtype,
            img_mean, img_std, norm_min, norm_max,
            files, image_dir, ensembles_full, use_novel)

    print_banner("SURVEY COMPLETE — all requested attacks finished.", width=100)


def parse_args():
    p = argparse.ArgumentParser(
        description="Survey-scale Florence-2 OCR robustness sweep "
                    "(FGSM/PGD/Patch x Tier1/Survey defense bank).")
    # GPU isolation (parsed early too, see _early_gpu_arg).
    p.add_argument("--gpu", type=int, default=-1,
                   help="GPU index to pin; <0 picks the freest GPU.")
    p.add_argument("--tier", choices=["tier1", "survey"], default="survey",
                   help="survey -> SURVEY ensembles; tier1 -> TIER1 ensembles.")
    p.add_argument("--attacks", nargs="+",
                   default=["fgsm", "pgd", "patch"],
                   choices=["fgsm", "pgd", "patch"],
                   help="Attacks to run sequentially in one process.")
    p.add_argument("--num-images", type=int, default=5000)

    # Base OCR script mirror.
    p.add_argument("--image-dir", type=str, default="Dataset/val2017")
    p.add_argument("--model-name", type=str, default="microsoft/Florence-2-base")
    p.add_argument("--model-revision", type=str, default="refs/pr/26")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--repetition-penalty", type=float, default=1.8)
    p.add_argument("--length-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--empty-cache-every", type=int, default=25)
    p.add_argument("--skip-clean-consistency", action="store_true",
                   help="Use fixed clean baseline=1.0 instead of a second "
                        "clean OCR pass.")

    # Attack params (FGSM eps=0.03; PGD eps=0.03/10 iters/alpha=eps/4;
    # Patch 35x35 center / 100 Adam steps / lr=0.02).
    p.add_argument("--eps", type=float, default=0.03)
    p.add_argument("--pgd-iters", type=int, default=10)
    p.add_argument("--patch-size", type=int, default=35)
    p.add_argument("--patch-iters", type=int, default=100)
    p.add_argument("--patch-lr", type=float, default=0.02)

    # Survey-specific.
    p.add_argument("--checkpoint-every", type=int, default=200,
                   help="Flush cadence (images) for state_{attack}.json.")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Pure in-memory run; do not read/write state files.")
    p.add_argument("--novel", action="store_true",
                   help="Add the sign_approx novel defense branch.")
    p.add_argument(
        "--output-dir", type=str,
        default="/home/king/Documents/Projects/Adversarial_Attack_Defence_Paper/"
                "results_survey_florence_ocr")
    return p.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
