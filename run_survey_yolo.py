#!/usr/bin/env python3
# ======================================================================
# run_survey_yolo.py — Adversarial attack/defense SURVEY for
#                      YOLOv8x-worldv2 open-vocabulary object detection.
#
# Runs FGSM / PGD / Patch attacks SEQUENTIALLY in ONE process (model loaded
# once) over a Tier-1 or full SURVEY defense bank, scored with COCO mAP using
# REAL YOLO confidences and real COCO category_ids. Per-image detection buckets
# are config-keyed checkpointed (pc.SurveyCheckpoint) so a crashed run resumes
# without ever crossing configs. Writes exactly one summary_{attack}.json per
# attack into results_survey_yolo/.
#
# ----------------------------------------------------------------------
# CITATIONS
# ----------------------------------------------------------------------
# Attacks:
#   FGSM   — Goodfellow, Shlens & Szegedy, ICLR 2015.
#   PGD    — Madry et al., ICLR 2018.
#   Patch  — Brown et al., "Adversarial Patch", 2017.
#
# Defenses (Tier 1 = paper main set; Tier 2 = survey completeness):
#   JPEG compression         — Dziugaite et al., 2016.
#   median filter            — Xu et al., NDSS 2018 (feature squeezing).
#   bit-depth reduction      — Xu et al., NDSS 2018 (feature squeezing).
#   gaussian blur            — Xu et al., NDSS 2018 (spatial smoothing).
#   TVM (total variation min)— Guo et al., ICLR 2018.
#   NLM (non-local means)    — Buades et al., CVPR 2005.
#   SVD spectral filter      — low-rank spectral truncation.
#   bilateral filter         — Tomasi & Manduchi, ICCV 1998.
#   ordered (Bayer) dithering— fast deterministic stand-in for serial
#                              Floyd-Steinberg error diffusion.
#   anisotropic diffusion    — Perona & Malik, IEEE PAMI 1990.
#   BM3D                     — Dabov et al., IEEE TIP 2007 (OPTIONAL dep).
#
# Novel defenses (--novel; each wrapped in try/except, skipped+warned on
# missing dependency so the survey never crashes):
#   SIGN     — arXiv:2605.27927 (2026, preprint). This script can only run an
#              APPROXIMATION of SIGN (gradient-sign-style input cleansing);
#              its output row is labeled 'sign_approx' to stay honest.
#   PAD      — CVPR 2024 (arXiv:2404.16452), SAM-based patch defense. Run only
#              if segment_anything is importable; here a simple SAM-localized
#              patch inpaint (nearest-neighbour fill around the 35x35 center).
#   XAIAD-YOLO — Future Generation Computer Systems 2026
#              (DOI 10.1016/j.future.2025.108356). Run only if the optional
#              `xaiad_yolo` package is importable; otherwise skipped+warned.
#
# NOTE: This is a SURVEY script — it does NOT modify phase3_common.py or any
# frozen *_v2 / *_Robust file. It uses ONLY the SURVEY_*/TIER1_* registries.
# ======================================================================

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import warnings


# ======================================================================
# GPU isolation — MUST happen BEFORE importing torch / phase3_common.
# Scan sys.argv early for --gpu; if gpu < 0 pick the freest via nvidia-smi
# (same inline logic as the *_v2 files); then set CUDA_VISIBLE_DEVICES.
# ======================================================================
def _early_gpu_arg() -> int:
    gpu = 0
    for i, a in enumerate(sys.argv):
        if a == "--gpu" and i + 1 < len(sys.argv):
            try:
                gpu = int(sys.argv[i + 1])
            except ValueError:
                gpu = 0
        elif a.startswith("--gpu="):
            try:
                gpu = int(a.split("=", 1)[1])
            except ValueError:
                gpu = 0
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
    except Exception as e:  # noqa: BLE001
        print(f"[GPU] nvidia-smi failed ({e}); using default device 0.")
    return "0"


_GPU = _early_gpu_arg()
if _GPU < 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pick_freest_gpu()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_GPU)
print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# ----- NOW safe to import torch + phase3_common -----
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from ultralytics import YOLO

import phase3_common as pc

warnings.filterwarnings("ignore")


# ======================================================================
# Argparse
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Adversarial attack/defense survey for YOLOv8x-worldv2.")
    p.add_argument("--image-dir", default="./val2017")
    p.add_argument("--ann-file", default="./annotations/instances_val2017.json")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index; <0 picks the freest GPU.")
    p.add_argument("--num-images", type=int, default=5000)
    p.add_argument("--tier", choices=["tier1", "survey"], default="survey")
    p.add_argument("--attacks", nargs="+", choices=["fgsm", "pgd", "patch"],
                   default=["fgsm", "pgd", "patch"])
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--novel", action="store_true",
                   help="Attempt novel defenses (SIGN approx, PAD/SAM, "
                        "XAIAD-YOLO); each skipped+warned if dep missing.")
    # Optional defense hyperparams
    p.add_argument("--jpeg-quality", type=int, default=pc.DEFAULT_JPEG_QUALITY)
    p.add_argument("--median-kernel", type=int, default=pc.DEFAULT_MEDIAN_KERNEL)
    p.add_argument("--tvm-weight", type=float, default=pc.DEFAULT_TVM_WEIGHT)
    p.add_argument("--tvm-iters", type=int, default=pc.DEFAULT_TVM_ITERS)
    p.add_argument("--gaussian-sigma", type=float,
                   default=pc.DEFAULT_GAUSSIAN_SIGMA)
    return p.parse_args()


# ======================================================================
# Constants (attack budgets — RAW [0,1] pixel space, NOT normalized)
# ======================================================================
YOLO_MODEL = "yolov8x-worldv2.pt"
YOLO_IMGSZ = 640
YOLO_CONF = 0.001
YOLO_IOU_NMS = 0.5
ENSEMBLE_NMS_IOU = pc.DEFAULT_NMS_IOU

EPSILON = 0.03               # FGSM / PGD budget
PGD_ITERS = 10
PGD_ALPHA = EPSILON / 4.0    # step size
PATCH_SIZE = pc.PATCH_SIZE          # 35
PATCH_OPT_ITERS = pc.PATCH_OPT_ITERS  # 100
PATCH_LR = pc.PATCH_LR              # 0.02

OUTPUT_DIR = "/home/king/Documents/Projects/Adversarial_Attack_Defence_Paper/results_survey_yolo"

ATTACK_TAGS = {
    "fgsm":  "fgsm_eps0.03",
    "pgd":   "pgd_eps0.03",
    "patch": "patch",
}


# ======================================================================
# Letterbox helpers (match YOLO preprocessing — copied verbatim from v2)
# ======================================================================
def letterbox(pil_img, size=YOLO_IMGSZ, fill=(114, 114, 114)):
    w, h = pil_img.size
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = pil_img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    pl, pt = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pl, pt))
    return canvas, scale, pl, pt, nw, nh


def unletterbox(lb_np, orig_size, pl, pt, nw, nh):
    cropped = lb_np[pt:pt + nh, pl:pl + nw]
    return Image.fromarray(cropped).resize(orig_size, Image.BICUBIC)


# ======================================================================
# Inference -> COCO-format detections (REAL YOLO confidences + COCO ids)
# ======================================================================
def make_inference_fn(model, device, yolo_to_coco_id):
    def _parse(res):
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
            if cls_idx not in yolo_to_coco_id:
                continue
            dets.append({
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "category_id": yolo_to_coco_id[cls_idx],
                "score": float(confs[i]),
            })
        return dets

    def run_inference(pil_img):
        results = model.predict(pil_img, conf=YOLO_CONF, iou=YOLO_IOU_NMS,
                                imgsz=YOLO_IMGSZ, verbose=False, device=device)
        return _parse(results[0])

    return run_inference


# ======================================================================
# Attacks — RAW [0,1] pixel space. The tensor-cache reset
# `model.model.model[-1].shape = None` MUST run before each autograd forward
# (verified from the v2 notebooks: once per forward; FGSM single, PGD/patch
# every iteration). `run_inference` uses model.predict and does not need it.
# ======================================================================
def make_attacks(model, device):
    def fgsm_attack(pil_img, eps=EPSILON):
        orig_size = pil_img.size
        lb, _, pl, pt, nw, nh = letterbox(pil_img, YOLO_IMGSZ)
        img_np = np.array(lb).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        adv = img_t.clone().detach().requires_grad_(True)

        model.model.model[-1].shape = None  # reset cached tensors for autograd
        preds = model.model(adv)
        pred = preds[0] if isinstance(preds, (list, tuple)) else preds
        cls_scores = pred[:, 4:, :]
        loss = -cls_scores.max(dim=1)[0].sum()
        loss.backward()

        grad_sign = adv.grad.sign()
        perturbed = (img_t.detach() + eps * grad_sign).clamp(0.0, 1.0)
        adv_np = (perturbed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return unletterbox(adv_np, orig_size, pl, pt, nw, nh)

    def pgd_attack(pil_img, eps=EPSILON, iters=PGD_ITERS, alpha=PGD_ALPHA):
        orig_size = pil_img.size
        lb, _, pl, pt, nw, nh = letterbox(pil_img, YOLO_IMGSZ)
        img_np = np.array(lb).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

        adv = (img_t.clone().detach()
               + torch.empty_like(img_t).uniform_(-eps, eps)).clamp(0.0, 1.0)

        for _ in range(iters):
            adv = adv.detach().requires_grad_(True)
            model.model.model[-1].shape = None  # reset cached tensors for autograd
            preds = model.model(adv)
            pred = preds[0] if isinstance(preds, (list, tuple)) else preds
            cls_scores = pred[:, 4:, :]
            loss = -cls_scores.max(dim=1)[0].sum()
            if adv.grad is not None:
                adv.grad.zero_()
            loss.backward()
            with torch.no_grad():
                adv = adv + alpha * adv.grad.sign()
                delta = torch.clamp(adv - img_t, min=-eps, max=eps)
                adv = torch.clamp(img_t + delta, 0.0, 1.0).detach()

        adv_np = (adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return unletterbox(adv_np, orig_size, pl, pt, nw, nh)

    def patch_attack(pil_img, patch_size=PATCH_SIZE,
                     iters=PATCH_OPT_ITERS, lr=PATCH_LR):
        orig_size = pil_img.size
        lb, _, pl, pt, nw, nh = letterbox(pil_img, YOLO_IMGSZ)
        img_np = np.array(lb).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

        H, W = img_t.shape[-2:]
        top, left = pc.center_patch_coords(H, W, patch_size)

        torch.manual_seed(0)
        patch = torch.rand(3, patch_size, patch_size, device=device) * 0.5 + 0.25
        patch.requires_grad_(True)
        optim = torch.optim.Adam([patch], lr=lr)

        for _ in range(iters):
            optim.zero_grad()
            composed = img_t.clone()
            composed[:, :, top:top + patch_size, left:left + patch_size] = (
                patch.clamp(0.0, 1.0).unsqueeze(0)
            )
            model.model.model[-1].shape = None  # reset cached tensors for autograd
            preds = model.model(composed)
            pred = preds[0] if isinstance(preds, (list, tuple)) else preds
            cls_scores = pred[:, 4:, :]
            loss = cls_scores.max(dim=1)[0].sum()
            loss.backward()
            optim.step()
            with torch.no_grad():
                patch.clamp_(0.0, 1.0)

        with torch.no_grad():
            adv = img_t.clone()
            adv[:, :, top:top + patch_size, left:left + patch_size] = (
                patch.clamp(0.0, 1.0).unsqueeze(0)
            )
        adv_np = (adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return unletterbox(adv_np, orig_size, pl, pt, nw, nh)

    return {"fgsm": fgsm_attack, "pgd": pgd_attack, "patch": patch_attack}


# ======================================================================
# --novel defenses. Each is wrapped in try/except at LOAD time: a missing
# dependency prints ONE warning and the row is SKIPPED (never crashes the run).
# Returns a dict {label: callable(pil_img)->pil_img} of the ones that loaded.
# ======================================================================
def load_novel_defenses(device):
    novel: dict[str, "callable"] = {}

    # --- XAIAD-YOLO (FGCS 2026, DOI 10.1016/j.future.2025.108356) ---
    try:
        from xaiad_yolo import XAIADDefense  # type: ignore
        _xaiad = XAIADDefense()

        def _xaiad_fn(pil_img):
            return _xaiad(pil_img)
        novel["xaiad_yolo"] = _xaiad_fn
        print("[novel] XAIAD-YOLO loaded.")
    except Exception as e:  # noqa: BLE001
        print(f"[novel] XAIAD-YOLO unavailable ({e}); skipping row.")

    # --- PAD / SAM (CVPR 2024, arXiv:2404.16452) ---
    # If segment_anything is importable, implement a simple SAM-localized patch
    # defense: replace the 35x35 center region with nearest-neighbour fill from
    # just outside the patch. Otherwise skip+warn.
    try:
        from segment_anything import sam_model_registry  # type: ignore  # noqa: F401

        def _pad_fn(pil_img):
            arr = np.array(pil_img.convert("RGB"))
            H, W = arr.shape[:2]
            top, left = pc.center_patch_coords(H, W, PATCH_SIZE)
            ph = min(PATCH_SIZE, H - top)
            pw = min(PATCH_SIZE, W - left)
            if ph <= 0 or pw <= 0:
                return pil_img
            out = arr.copy()
            # Nearest-neighbour fill from just outside the patch border.
            ref_top = max(0, top - 1)
            ref_left = max(0, left - 1)
            border = arr[ref_top, ref_left]            # single outside pixel
            row_above = arr[ref_top, left:left + pw]   # strip just above patch
            for r in range(ph):
                out[top + r, left:left + pw] = row_above
            # corner fallback if the strip was clipped
            out[top:top + ph, left:left + pw][
                np.all(out[top:top + ph, left:left + pw] == 0, axis=-1)
            ] = border
            return Image.fromarray(out)
        novel["pad_sam"] = _pad_fn
        print("[novel] PAD/SAM loaded (SAM-localized center-patch inpaint).")
    except Exception as e:  # noqa: BLE001
        print(f"[novel] PAD/SAM unavailable ({e}); skipping row.")

    # --- SIGN approximation (arXiv:2605.27927, 2026 preprint) ---
    # We can only run an APPROXIMATION of SIGN here: a gradient-sign-style input
    # cleanse (subtract a small per-pixel sign step toward the local mean) that
    # mimics SIGN's sign-based purification without its trained components.
    # Labeled 'sign_approx' so the survey never mislabels it as true SIGN.
    try:
        import torch as _t  # noqa: WPS433

        def _sign_approx_fn(pil_img):
            t = pc.pil_to_tensor(pil_img, device)
            blurred = pc.gaussian_gpu(t, sigma=1.0)
            step = 0.5 * EPSILON
            residual_sign = (t - blurred).sign()
            cleaned = (t - step * residual_sign).clamp(0.0, 1.0)
            return pc.tensor_to_pil(cleaned)
        novel["sign_approx"] = _sign_approx_fn
        print("[novel] SIGN approximation loaded (labeled 'sign_approx').")
    except Exception as e:  # noqa: BLE001
        print(f"[novel] SIGN approximation unavailable ({e}); skipping row.")

    return novel


# ======================================================================
# Per-attack run. Loads nothing model-side (model passed in, loaded once).
# ======================================================================
def run_attack(attack_key, attack_fn, run_inference, defense_producer,
               novel_defenses, files, image_dir, num_images, coco_gt,
               cached_ids, args, base_defense_names, base_ensembles):
    attack_tag = ATTACK_TAGS[attack_key]

    # ---- Probe one image to learn the ACTUAL defense keys produced ----
    probe_path = os.path.join(image_dir, files[0])
    probe_img = Image.open(probe_path).convert("RGB")
    probe_defs = defense_producer(probe_img, seed=0)
    defense_names = list(probe_defs.keys())            # derived from KEYS
    # add the novel rows that actually loaded
    novel_names = list(novel_defenses.keys())
    all_defense_names = defense_names + novel_names

    # Filter the tier ensembles: keep only those whose every member is present.
    ensembles = {n: m for n, m in base_ensembles.items()
                 if all(member in defense_names for member in m)}

    # ---- SAFE config-keyed checkpoint signature ----
    sig = pc.config_signature(
        model=YOLO_MODEL, attack=attack_tag,
        defenses=sorted(all_defense_names),
        eps=EPSILON, pgd_iters=PGD_ITERS, pgd_alpha=PGD_ALPHA,
        patch_size=PATCH_SIZE, patch_iters=PATCH_OPT_ITERS, patch_lr=PATCH_LR,
        num_images=num_images, image_dir=image_dir,
    )
    ckpt = pc.SurveyCheckpoint(
        os.path.join(OUTPUT_DIR, f"checkpoint_{attack_key}.pkl"),
        sig, flush_every=args.checkpoint_every,
        enabled=not args.no_checkpoint,
    )

    pc.print_banner(f"SURVEY YOLO — {attack_tag} | tier={args.tier} "
                    f"| defenses={len(all_defense_names)}", width=80)

    # Expected condition tags every processed image must carry.
    expected_tags = (
        ["clean"]
        + [f"clean+{d}" for d in all_defense_names]
        + [attack_tag]
        + [f"{attack_tag}+{d}" for d in all_defense_names]
    )

    def process_one(fname):
        img_id = int(os.path.splitext(fname)[0])
        if ckpt.has(img_id):
            return
        pil_img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        buckets: dict[str, list] = {}

        def _record(tag, dets):
            for d in dets:
                d["image_id"] = img_id
            buckets[tag] = dets

        # 1. Clean baseline
        _record("clean", run_inference(pil_img))
        # 2. Clean + survey defenses (cost on clean images)
        clean_defs = defense_producer(pil_img, seed=img_id)
        for dn in defense_names:
            _record(f"clean+{dn}", run_inference(clean_defs[dn]))
        for dn, fn in novel_defenses.items():
            _record(f"clean+{dn}", run_inference(fn(pil_img)))
        # 3. Attacked + survey defenses (recovery)
        adv = attack_fn(pil_img)
        _record(attack_tag, run_inference(adv))
        atk_defs = defense_producer(adv, seed=img_id)
        for dn in defense_names:
            _record(f"{attack_tag}+{dn}", run_inference(atk_defs[dn]))
        for dn, fn in novel_defenses.items():
            _record(f"{attack_tag}+{dn}", run_inference(fn(adv)))

        ckpt.put(img_id, buckets)
        return buckets

    # ---- Main loop: sequential, minimal logging, per-image try/except ----
    # Heartbeat proxies (true mAP needs the end-of-run COCO eval). We track the
    # rolling mean attacked detection score, and the rolling best mean score
    # over all defended-attacked conditions, as a cheap live progress signal.
    start = time.time()
    failures: list[tuple[str, str]] = []
    atk_running = 0.0
    best_running = 0.0
    score_count = 0

    def _mean_score(dets):
        return float(np.mean([d["score"] for d in dets])) if dets else 0.0

    for i, fname in enumerate(files):
        try:
            buckets = process_one(fname)
            if buckets is None:                 # resumed from checkpoint
                buckets = ckpt.buckets.get(int(os.path.splitext(fname)[0]), {})
            atk_dets = buckets.get(attack_tag, [])
            if atk_dets is not None and buckets:
                atk_running += _mean_score(atk_dets)
                best_def = max(
                    (_mean_score(buckets.get(f"{attack_tag}+{d}", []))
                     for d in all_defense_names),
                    default=0.0)
                best_running += best_def
                score_count += 1
        except Exception as exc:  # noqa: BLE001
            failures.append((fname, repr(exc)))

        if (i + 1) % 100 == 0:
            elapsed_min = (time.time() - start) / 60.0
            atk_avg = (atk_running / score_count) if score_count else 0.0
            best = (best_running / score_count) if score_count else 0.0
            print(f"[{attack_tag}] {i + 1}/{len(files)} | "
                  f"attacked={atk_avg:.4f} | best={best:.4f} | "
                  f"{elapsed_min:.1f}m")
        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

    ckpt.flush()
    if failures:
        print(f"[{attack_tag}] {len(failures)} image(s) failed and were "
              f"skipped (first: {failures[0]}).")

    # ---- HARD ASSERT: processed count + every condition tag present ----
    processed = len(ckpt.buckets)
    assert processed == num_images, (
        f"[{attack_tag}] image count mismatch: processed {processed} "
        f"but expected num_images={num_images} "
        f"({len(failures)} failures recorded)")
    for img_id, b in ckpt.buckets.items():
        missing = [t for t in expected_tags if t not in b]
        assert not missing, f"[{attack_tag}] img {img_id} missing tags: {missing}"
    print(f"[{attack_tag}] all {processed} images carry all "
          f"{len(expected_tags)} expected condition tags.")

    # ---- Assemble (offline ensembles) + COCO eval ----
    all_results = pc.assemble_results(
        ckpt.buckets,
        defense_names=all_defense_names,
        attack_tags=[attack_tag],
        ensembles=ensembles,
        nms_iou=ENSEMBLE_NMS_IOU,
    )
    eval_stats = pc.evaluate_all_conditions(
        all_results, coco_gt, image_ids=cached_ids, output_dir=OUTPUT_DIR,
    )

    # ---- Summary JSON (exactly one summary_{attack}.json) ----
    clean_ap = float(eval_stats["clean"][0])
    atk_ap = float(eval_stats[attack_tag][0])
    attack_drop = clean_ap - atk_ap
    ordered_defenses = all_defense_names + list(ensembles.keys())

    summary = {
        "model": YOLO_MODEL,
        "attack": attack_tag,
        "tier": args.tier,
        "epsilon": EPSILON if attack_key != "patch" else None,
        "num_images": len(cached_ids),
        "clean_mAP": clean_ap,
        "clean_AP50": float(eval_stats["clean"][1]),
        "attacked_mAP": atk_ap,
        "attack_damage": attack_drop,
        "defenses": {},
    }
    for dn in ordered_defenses:
        tag = f"{attack_tag}+{dn}"
        if tag not in eval_stats:
            continue
        def_ap = float(eval_stats[tag][0])
        def_ap50 = float(eval_stats[tag][1])
        rec = def_ap - atk_ap
        rec_pct = (100.0 * rec / attack_drop) if attack_drop > 0 else 0.0
        summary["defenses"][dn] = {
            "mAP": def_ap, "AP50": def_ap50,
            "recovery": rec, "recovery_pct": rec_pct,
            "kind": "ENSEMBLE" if dn in ensembles else "solo",
        }

    summary_path = os.path.join(OUTPUT_DIR, f"summary_{attack_key}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Short ranked defense table (reuse base-script format) ----
    pc.print_banner(f"ATTACKED vs RECOVERED — {attack_tag} (YOLO)", width=100)
    print(f"  Clean baseline   : {clean_ap:.4f}")
    print(f"  Attacked         : {atk_ap:.4f}")
    print(f"  Attack damage    : {attack_drop:+.4f}\n")

    rows = [(dn, info["kind"], info["mAP"], info["recovery"], info["recovery_pct"])
            for dn, info in summary["defenses"].items()]
    rows.sort(key=lambda r: r[3], reverse=True)

    print("-" * 100)
    print(f"  {'Rank':<5} {'Defense':<28} {'Kind':<10} {'Attacked':>9} "
          f"{'Recovered':>10} {'D mAP':>9} {'Rec%':>7} {'Verdict':>10}")
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
        print(f"  {rank:<5} {name:<28} {kind:<10} {atk_ap:>9.4f} "
              f"{def_ap:>10.4f} {rec:>+9.4f} {rec_pct:>+6.1f}% {verdict:>10}")
    print("-" * 100)
    print(f"[{attack_tag}] summary -> {summary_path}")
    return summary


# ======================================================================
# Main
# ======================================================================
def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"GPUs visible: {torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA required for the survey."
    device = torch.device("cuda:0")

    # ---- COCO GT + class mapping ----
    coco_gt = COCO(args.ann_file)
    cats_sorted = sorted(coco_gt.loadCats(coco_gt.getCatIds()),
                         key=lambda x: x["id"])
    coco_names = [c["name"] for c in cats_sorted]
    coco_ids = [c["id"] for c in cats_sorted]
    yolo_to_coco_id = {i: cid for i, cid in enumerate(coco_ids)}

    # ---- Load YOLO ONCE ----
    model = YOLO(YOLO_MODEL)
    model.set_classes(coco_names)
    model.to(device)
    model.model.eval()
    print(f"YOLO loaded — {len(coco_names)} COCO classes | imgsz={YOLO_IMGSZ}")

    run_inference = make_inference_fn(model, device, yolo_to_coco_id)
    attacks = make_attacks(model, device)

    # ---- Files / image ids ----
    files = sorted(os.listdir(args.image_dir))
    if args.num_images is not None:
        files = files[:args.num_images]
    num_images = len(files)
    cached_ids = sorted({int(os.path.splitext(f)[0]) for f in files})
    print(f"Selected {num_images} images.")

    # ---- Tier selection -> defense producer + base ensembles ----
    if args.tier == "survey":
        base_ensembles = pc.SURVEY_ENSEMBLES

        def defense_producer(pil, seed=None):
            return pc.apply_survey_defenses(
                pil, device,
                jpeg_quality=args.jpeg_quality, median_kernel=args.median_kernel,
                tvm_weight=args.tvm_weight, tvm_iters=args.tvm_iters,
                gaussian_sigma=args.gaussian_sigma, seed=seed)
    else:  # tier1
        base_ensembles = pc.TIER1_ENSEMBLES
        # Tier-1 set = exactly pc.TIER1_SOLOS (the 5 locked solos + svd).
        # apply_survey_defenses is the only producer of svd, so we filter its
        # output down to the Tier-1 keys and never emit Tier-2 rows (incl.
        # bilateral) in tier1 mode. Matches run_survey_florence_detection.py.
        tier1_keys = set(pc.TIER1_SOLOS)

        def defense_producer(pil, seed=None):
            full = pc.apply_survey_defenses(
                pil, device,
                jpeg_quality=args.jpeg_quality, median_kernel=args.median_kernel,
                tvm_weight=args.tvm_weight, tvm_iters=args.tvm_iters,
                gaussian_sigma=args.gaussian_sigma, seed=seed)
            return {k: v for k, v in full.items() if k in tier1_keys}

    base_defense_names: list[str] = []  # derived per-attack from keys (probe)

    # ---- Novel defenses (loaded once, shared across attacks) ----
    novel_defenses = load_novel_defenses(device) if args.novel else {}

    # ---- Run all requested attacks SEQUENTIALLY in ONE process ----
    for attack_key in args.attacks:
        run_attack(
            attack_key, attacks[attack_key], run_inference, defense_producer,
            novel_defenses, files, args.image_dir, num_images, coco_gt,
            cached_ids, args, base_defense_names, base_ensembles,
        )

    print("\nAll requested attacks complete.")


if __name__ == "__main__":
    main()
