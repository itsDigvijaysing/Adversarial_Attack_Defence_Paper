#!/usr/bin/env python3
"""
Robust Patch-Attack OCR benchmark on Florence-2 with post-attack defenses.

What this script does:
1. Runs Florence-2 OCR on clean images.
2. Attacks images with optimized adversarial patch.
3. Applies solo and ensemble defenses on attacked images.
4. Prints an "ATTACKED vs RECOVERED" ranked table in terminal.
5. Saves resumable state, summary JSON, and failure logs.
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from io import BytesIO
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageFilter
from skimage.restoration import denoise_tv_chambolle
from transformers import AutoModelForCausalLM, AutoProcessor


def print_banner(title: str, char: str = "=", width: int = 100) -> None:
    print(char * width)
    print(f"{title:^{width}}")
    print(char * width)


def log(message: str, fh=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


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


def defend_jpeg(pil_img: Image.Image, quality: int) -> Image.Image:
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def defend_median(pil_img: Image.Image, kernel: int) -> Image.Image:
    return pil_img.filter(ImageFilter.MedianFilter(size=kernel))


def defend_gaussian(pil_img: Image.Image, sigma: float) -> Image.Image:
    return pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))


def defend_tvm(pil_img: Image.Image, weight: float) -> Image.Image:
    arr = np.asarray(pil_img).astype(np.float64) / 255.0
    denoised = denoise_tv_chambolle(arr, weight=weight, channel_axis=-1)
    denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(denoised)


def defend_blur_tvm(pil_img: Image.Image, sigma: float, weight: float) -> Image.Image:
    return defend_tvm(defend_gaussian(pil_img, sigma=sigma), weight=weight)


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

    best_idx = max(range(len(texts)), key=lambda i: (pairwise[i], len(norm[i]), len(texts[i])))
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
            parsed = processor.post_process_generation(raw, task="<OCR>", image_size=(img_w, img_h))
            return decode_ocr(parsed)
        except Exception:
            return sanitize_generated_text(raw)


def patch_attack_ocr(
    model,
    processor,
    pil_img: Image.Image,
    patch_size: int,
    patch_iters: int,
    patch_lr: float,
    random_location: bool,
    device,
    torch_dtype,
    max_new_tokens: int,
    num_beams: int,
    repetition_penalty: float,
    length_penalty: float,
    img_mean: torch.Tensor,
    img_std: torch.Tensor,
    norm_min: torch.Tensor,
    norm_max: torch.Tensor,
) -> Image.Image:
    orig_size = pil_img.size
    inputs = processor(text="<OCR>", images=pil_img, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype).detach()

    with torch.no_grad():
        target_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=repetition_penalty,
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

        outputs = model(input_ids=input_ids, pixel_values=x_adv, labels=target_ids)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError("Patch attack loss became non-finite.")

        # Maximize task loss to degrade OCR output.
        (-loss).backward()
        if patch_param.grad is None:
            raise RuntimeError("Patch attack gradient is None.")
        optimizer.step()

    with torch.no_grad():
        patch_img = patch_param.clamp(0.0, 1.0)
        patch_norm = (patch_img - img_mean32) / img_std32
        x_adv32 = x_orig.clone()
        x_adv32[:, :, top:top + ps, left:left + ps] = patch_norm
        x_adv32 = torch.max(torch.min(x_adv32, norm_max32), norm_min32)

    adv_denorm = x_adv32 * img_std32 + img_mean32
    adv_denorm = torch.clamp(adv_denorm, 0.0, 1.0)
    adv_np = (adv_denorm.squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy() * 255).astype(np.uint8)

    adv_pil = Image.fromarray(adv_np)
    if adv_pil.size != orig_size:
        adv_pil = adv_pil.resize(orig_size, Image.BICUBIC)
    return adv_pil


def make_default_state(defense_names: List[str]) -> Dict:
    return {
        "processed": [],
        "failed": [],
        "clean_baseline_sum": 0.0,
        "attacked_sim_sum": 0.0,
        "defenses": {
            name: {"sim_sum": 0.0, "recovery_sum": 0.0}
            for name in defense_names
        },
    }


def load_state(state_path: str, defense_names: List[str]) -> Dict:
    if not os.path.isfile(state_path):
        return make_default_state(defense_names)

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    for key in ["processed", "failed", "clean_baseline_sum", "attacked_sim_sum", "defenses"]:
        if key not in state:
            raise RuntimeError(f"State file is missing key: {key}")

    for name in defense_names:
        if name not in state["defenses"]:
            state["defenses"][name] = {"sim_sum": 0.0, "recovery_sum": 0.0}

    return state


def save_state(state_path: str, state: Dict) -> None:
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, state_path)


def resolve_image_dir(primary_dir: str) -> str:
    candidates = [
        primary_dir,
        "dataset/val2017",
        "./dataset/val2017",
        "val2017",
        "./val2017",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise RuntimeError(
        "Could not find image directory. Tried: " + ", ".join(candidates)
    )


def verdict_from_recovery(recovery: float) -> str:
    if recovery > 0.005:
        return "RECOVERS"
    if recovery > 0.0:
        return "marginal"
    if recovery > -0.005:
        return "neutral"
    return "HURTS"


def run(args) -> None:
    set_seed(args.seed)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    summary_path = os.path.join(output_dir, "summary.json")
    failed_path = os.path.join(output_dir, "failed_images.json")
    state_path = args.state_path or os.path.join(output_dir, "state.json")

    log_mode = "a" if os.path.isfile(log_path) else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_f:
        if args.expected_conda_env:
            current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
            if current_env != args.expected_conda_env:
                msg = (
                    f"Expected conda env '{args.expected_conda_env}', "
                    f"current env is '{current_env or 'unknown'}'."
                )
                if args.strict_env:
                    raise RuntimeError(msg)
                log(f"[WARN] {msg}", log_f)

        image_dir = resolve_image_dir(args.image_dir)
        files = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        )
        if args.num_images is not None:
            files = files[:args.num_images]
        if not files:
            raise RuntimeError(f"No images found in {image_dir}")

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch_dtype = torch.float16
        else:
            device = torch.device("cpu")
            torch_dtype = torch.float32

        if args.patch_size < 1:
            raise ValueError("--patch-size must be >= 1")
        if args.patch_iters < 1:
            raise ValueError("--patch-iters must be >= 1")
        if args.patch_lr <= 0:
            raise ValueError("--patch-lr must be > 0")

        log("Starting robust Patch-Attack OCR recovery run", log_f)
        log(f"Image dir: {image_dir}", log_f)
        log(f"Images requested: {len(files)}", log_f)
        log(f"Device: {device}, dtype: {torch_dtype}", log_f)
        log(f"Model: {args.model_name} @ {args.model_revision}", log_f)
        log(
            f"Patch params: size={args.patch_size}, iters={args.patch_iters}, "
            f"lr={args.patch_lr}, random_location={not args.no_random_location}",
            log_f,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            revision=args.model_revision,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        processor = AutoProcessor.from_pretrained(
            args.model_name,
            revision=args.model_revision,
            trust_remote_code=True,
        )

        img_mean = torch.tensor(
            processor.image_processor.image_mean,
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        img_std = torch.tensor(
            processor.image_processor.image_std,
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        norm_min = (0.0 - img_mean) / img_std
        norm_max = (1.0 - img_mean) / img_std

        branches = {
            "jpeg": lambda im: defend_jpeg(im, quality=args.jpeg_quality),
            "median": lambda im: defend_median(im, kernel=args.median_kernel),
            "gaussian": lambda im: defend_gaussian(im, sigma=args.gaussian_sigma),
            "tvm": lambda im: defend_tvm(im, weight=args.tvm_weight),
            "blur_tvm": lambda im: defend_blur_tvm(im, sigma=args.gaussian_sigma, weight=args.tvm_weight),
        }

        ensembles = {
            "ens_blur_tvm_combo": ["jpeg", "blur_tvm", "median"],
            "ens_jpeg_median_gaussian": ["jpeg", "median", "gaussian"],
            "ens_jpeg_median_tvm": ["jpeg", "median", "tvm"],
        }

        all_defense_names = list(branches.keys()) + list(ensembles.keys())

        if args.reset_state and os.path.isfile(state_path):
            os.remove(state_path)
            log(f"Reset existing state: {state_path}", log_f)

        state = load_state(state_path, all_defense_names)
        processed_set = set(state["processed"])
        pending = [f for f in files if f not in processed_set]

        if processed_set:
            log(f"Resuming with {len(processed_set)} already processed images.", log_f)
        log(f"Pending images this run: {len(pending)}", log_f)

        gen_cfg = {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "repetition_penalty": args.repetition_penalty,
            "length_penalty": args.length_penalty,
        }

        start = time.time()
        for i, fname in enumerate(pending, 1):
            path = os.path.join(image_dir, fname)
            try:
                clean_img = Image.open(path).convert("RGB")

                clean_ref = get_ocr_text(
                    model, processor, clean_img, device, torch_dtype,
                    **gen_cfg,
                )

                if args.skip_clean_consistency:
                    clean_baseline = 1.0
                else:
                    clean_again = get_ocr_text(
                        model, processor, clean_img, device, torch_dtype,
                        **gen_cfg,
                    )
                    clean_baseline = text_similarity(clean_ref, clean_again)

                adv_img = patch_attack_ocr(
                    model, processor, clean_img,
                    patch_size=args.patch_size,
                    patch_iters=args.patch_iters,
                    patch_lr=args.patch_lr,
                    random_location=not args.no_random_location,
                    device=device,
                    torch_dtype=torch_dtype,
                    img_mean=img_mean,
                    img_std=img_std,
                    norm_min=norm_min,
                    norm_max=norm_max,
                    **gen_cfg,
                )

                attacked_text = get_ocr_text(
                    model, processor, adv_img, device, torch_dtype,
                    **gen_cfg,
                )
                attacked_sim = text_similarity(clean_ref, attacked_text)

                defense_sim = {}
                branch_text = {}

                for name, fn in branches.items():
                    defended_img = fn(adv_img)
                    defended_txt = get_ocr_text(
                        model, processor, defended_img, device, torch_dtype,
                        **gen_cfg,
                    )
                    branch_text[name] = defended_txt
                    defense_sim[name] = text_similarity(clean_ref, defended_txt)

                for ens_name, members in ensembles.items():
                    voted = consensus_vote_text([branch_text[m] for m in members])
                    defense_sim[ens_name] = text_similarity(clean_ref, voted)

                state["clean_baseline_sum"] += clean_baseline
                state["attacked_sim_sum"] += attacked_sim
                for name, sim in defense_sim.items():
                    state["defenses"][name]["sim_sum"] += sim
                    state["defenses"][name]["recovery_sum"] += (sim - attacked_sim)

                state["processed"].append(fname)

            except Exception as exc:
                msg = f"{fname}: {exc}"
                state["failed"].append(msg)
                log(f"[WARN] Skipped image due to error: {msg}", log_f)

            if i % args.flush_every == 0 or i == len(pending):
                save_state(state_path, state)

            if i % args.log_every == 0 or i == len(pending):
                done = len(state["processed"])
                atk_avg = state["attacked_sim_sum"] / done if done > 0 else 0.0
                elapsed = time.time() - start
                log(
                    f"Progress {i}/{len(pending)} this run | total_done={done} "
                    f"| attacked_avg={atk_avg:.4f} | elapsed={elapsed/60:.1f} min",
                    log_f,
                )

            if torch.cuda.is_available() and args.empty_cache_every > 0 and i % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

        n = len(state["processed"])
        if n == 0:
            raise RuntimeError("No successful images processed. Check failed_images.json for details.")

        clean_avg = state["clean_baseline_sum"] / n
        attacked_avg = state["attacked_sim_sum"] / n
        attack_drop = clean_avg - attacked_avg

        rows = []
        for name in all_defense_names:
            sim_avg = state["defenses"][name]["sim_sum"] / n
            rec_avg = state["defenses"][name]["recovery_sum"] / n
            rec_pct = (100.0 * rec_avg / attack_drop) if attack_drop > 1e-12 else 0.0
            kind = "ENSEMBLE" if name in ensembles else "solo"
            rows.append({
                "name": name,
                "kind": kind,
                "attacked": attacked_avg,
                "recovered": sim_avg,
                "delta": rec_avg,
                "recovery_pct": rec_pct,
                "verdict": verdict_from_recovery(rec_avg),
            })

        rows.sort(key=lambda x: x["delta"], reverse=True)

        attack_tag = f"patch_ps{args.patch_size}_i{args.patch_iters}"
        summary = {
            "model": args.model_name,
            "task": "OCR",
            "attack": attack_tag,
            "patch_size": args.patch_size,
            "patch_iters": args.patch_iters,
            "patch_lr": args.patch_lr,
            "patch_random_location": not args.no_random_location,
            "images": n,
            "clean_baseline": clean_avg,
            "attacked": attacked_avg,
            "attack_damage": attack_drop,
            "ranked_defenses": rows,
            "failed_images": len(state["failed"]),
            "config": {
                "image_dir": image_dir,
                "num_images_requested": args.num_images,
                "jpeg_quality": args.jpeg_quality,
                "gaussian_sigma": args.gaussian_sigma,
                "tvm_weight": args.tvm_weight,
                "median_kernel": args.median_kernel,
                "patch_size": args.patch_size,
                "patch_iters": args.patch_iters,
                "patch_lr": args.patch_lr,
                "patch_random_location": not args.no_random_location,
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
                "repetition_penalty": args.repetition_penalty,
                "length_penalty": args.length_penalty,
            },
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(state["failed"], f, indent=2)

        print()
        print_banner(f"ATTACKED vs RECOVERED - {attack_tag} (FLORENCE-2 OCR)", width=100)
        print(f"  Clean baseline   : {clean_avg:.4f}")
        print(f"  Attacked         : {attacked_avg:.4f}")
        print(f"  Attack damage    : {attack_drop:+.4f}\n")

        print("-" * 100)
        print(
            f"  {'Rank':<5} {'Defense':<28} {'Kind':<10} {'Attacked':>9} "
            f"{'Recovered':>10} {'DeltaSim':>9} {'Rec%':>7} {'Verdict':>10}"
        )
        print("-" * 100)
        for rank, row in enumerate(rows, 1):
            print(
                f"  {rank:<5} {row['name']:<28} {row['kind']:<10} {row['attacked']:>9.4f} "
                f"{row['recovered']:>10.4f} {row['delta']:>+9.4f} "
                f"{row['recovery_pct']:>+6.1f}% {row['verdict']:>10}"
            )
        print("-" * 100)

        best = rows[0]
        worst = rows[-1]
        print(
            f"\n  >> BEST  : {best['name']} ({best['kind']})  "
            f"sim {attacked_avg:.4f} -> {best['recovered']:.4f}  "
            f"({best['delta']:+.4f}, {best['recovery_pct']:+.1f}%)"
        )
        print(
            f"  >> WORST : {worst['name']} ({worst['kind']})  "
            f"sim {attacked_avg:.4f} -> {worst['recovered']:.4f}  ({worst['delta']:+.4f})"
        )

        solo_rec = [r["delta"] for r in rows if r["kind"] == "solo"]
        ens_rec = [r["delta"] for r in rows if r["kind"] == "ENSEMBLE"]
        if solo_rec and ens_rec:
            print(
                f"\n  Solo     recovery  avg={np.mean(solo_rec):+.4f}  "
                f"max={max(solo_rec):+.4f}  min={min(solo_rec):+.4f}"
            )
            print(
                f"  Ensemble recovery  avg={np.mean(ens_rec):+.4f}  "
                f"max={max(ens_rec):+.4f}  min={min(ens_rec):+.4f}"
            )
            winner = "ENSEMBLING" if max(ens_rec) > max(solo_rec) else "SOLO"
            print(f"  >> Best-of-kind winner: {winner}")

        log(f"Saved summary: {summary_path}", log_f)
        log(f"Saved state: {state_path}", log_f)
        log(f"Saved failed-image log: {failed_path}", log_f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Robust Patch-Attack OCR defense benchmark for Florence-2-base",
    )
    parser.add_argument("--image-dir", type=str, default="dataset/val2017")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--patch-size", type=int, default=35)
    parser.add_argument("--patch-iters", type=int, default=100)
    parser.add_argument("--patch-lr", type=float, default=0.02)
    parser.add_argument(
        "--no-random-location",
        action="store_true",
        help="Use center patch instead of random patch position.",
    )

    parser.add_argument("--model-name", type=str, default="microsoft/Florence-2-base")
    parser.add_argument("--model-revision", type=str, default="refs/pr/26")

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--repetition-penalty", type=float, default=1.8)
    parser.add_argument("--length-penalty", type=float, default=1.0)

    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--tvm-weight", type=float, default=0.05)
    parser.add_argument("--median-kernel", type=int, default=3)

    parser.add_argument("--output-dir", type=str, default="results_patch_florence2_ocr_robust")
    parser.add_argument("--state-path", type=str, default="")
    parser.add_argument("--reset-state", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--empty-cache-every", type=int, default=25)

    parser.add_argument("--expected-conda-env", type=str, default="vlm_ftune")
    parser.add_argument("--strict-env", action="store_true")
    parser.add_argument(
        "--skip-clean-consistency",
        action="store_true",
        help="Use fixed clean baseline=1.0 instead of second clean OCR pass.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
