#!/usr/bin/env python3
# ======================================================================
# p4_owl.py — the OWL-ViT entry point for Phase 4 (the fourth model).
#
#   python phase4/p4_owl.py --experiment oblivious_survey --n 1000 --gpu 0
#   python phase4/p4_owl.py --experiment adaptive_bpda --target median --n 300
#   python phase4/p4_owl.py --preflight
#   python phase4/p4_owl.py --experiment oblivious_survey --smoke-test
#
# MODEL. google/owlvit-base-patch32 (~150M parameters): an open-vocabulary
# detector built on CLIP ViT-B/32. Every COCO class is a text query
# ("a photo of a <class>"); each of the 24x24 = 576 image patches predicts one
# box and a logit per query. Unlike Florence-2 it has real confidences (the
# sigmoid of the logit), so its mAP is confidence-ranked like YOLO's.
#
# WHAT IS SHARED. Defenses, ensembles and the NMS merge are phase3_common's
# (through p4_core), the PGD loop, seeded init, quantization, BPDA surrogates
# and the union/mean EOT aggregation are p4_yolo's, and the native-resolution
# attack frame is p4_florence's. Nothing in those files is changed.
#
# TWO EXPERIMENTS
#   oblivious_survey : clean and oblivious-PGD images through EVERY transform
#       and the 7 ensembles (the Phase-3 style table: clean cost + recovery).
#   adaptive_bpda    : oblivious PGD vs PGD through --target (BPDA + EOT), the
#       same experiment as p4_yolo / p4_florence.
#
# OBJECTIVE (both attacks). As for YOLO: J = sum over patches of the max-class
# confidence (sigmoid of the logit); PGD lowers J, i.e. suppresses every
# detection. With --eot-loss max the per-patch max is taken over the branches
# (the NMS union keeps a box if ANY branch keeps it); patches line up across
# branches because every branch is resized to the same 768x768 input.
# ======================================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p4_core as core                                     # noqa: E402
import p4_yolo as ya              # noqa: E402  (shared attack machinery only)
import p4_florence as fl          # noqa: E402  (native frame, preprocessing, HF cache)
from p4_core import rt                                     # noqa: E402

EXPERIMENTS = ("oblivious_survey", "adaptive_bpda")
# OWL-ViT always attacks at native resolution (as Florence does).
STEM_FLAG_DEFAULTS = dict(ya.STEM_FLAG_DEFAULTS, attack_space="native")
TRACK = "owl"
# The survey transforms scored on detection besides the 7 base branches, as in
# clean_control --all-transforms for YOLO (BM3D and random resize-and-pad are
# OCR-only there: random resize never maps boxes back).
SURVEY_EXTRA = ("bit_depth", "dithering", "nlm", "anisotropic")
OBL, ADP, META = ya.OBLIVIOUS_TAG, ya.ADAPTIVE_TAG, ya.META_KEY


# ======================================================================
# 1. The model.
# ======================================================================
class OwlAdapter:
    """Loads OWL-ViT once; the text queries are embedded once and cached."""

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model_name = args.owl_model
        self.model = None
        self.processor = None
        self.query_embeds = None       # [1, Q, D]
        self.query_mask = None         # [1, Q]
        self.cat_ids = []              # query index -> COCO category id
        self.geom = None

    def sig_fields(self, args):
        return {"owl_model": args.owl_model, "prompt": args.prompt,
                "conf": args.conf, "max_det": args.max_det,
                "owl_nms_iou": args.nms_iou, "dtype": "float32"}

    def load(self, coco_gt):
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        torch = rt.torch
        self.processor = OwlViTProcessor.from_pretrained(self.model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(
            self.model_name, torch_dtype=torch.float32).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cats = sorted(coco_gt.loadCats(coco_gt.getCatIds()),
                      key=lambda c: c["id"])
        self.cat_ids = [c["id"] for c in cats]
        prompts = [self.args.prompt.format(c["name"]) for c in cats]
        text = self.processor(text=[prompts], return_tensors="pt")
        self.input_ids = text["input_ids"].to(self.device)
        self.attention_mask = text["attention_mask"].to(self.device)
        self.geom = fl.processor_geometry(self.processor)
        print(f"[owl] loaded {self.model_name} | float32 | device={self.device}"
              f" | {len(prompts)} queries ('{prompts[0]}', ...) | input "
              f"{self.geom[0]}x{self.geom[1]}")

    def _embed_queries(self, pixel_values):
        """Text embeddings exactly as the model's own forward computes them."""
        torch = rt.torch
        with torch.no_grad():
            q, _, _ = self.model.image_text_embedder(
                input_ids=self.input_ids, pixel_values=pixel_values,
                attention_mask=self.attention_mask)
        Q = self.input_ids.shape[0]
        self.query_embeds = q.reshape(1, Q, q.shape[-1]).detach()
        self.query_mask = (self.input_ids[..., 0] > 0).reshape(1, Q)

    def heads(self, pixel_values, boxes=True):
        """(logits [B,P,Q], boxes [B,P,4] cxcywh in [0,1] or None)."""
        torch = rt.torch
        feature_map, _ = self.model.image_embedder(pixel_values=pixel_values)
        B, h, w, d = feature_map.shape
        feats = torch.reshape(feature_map, (B, h * w, d))
        q = self.query_embeds.expand(B, -1, -1)
        qm = self.query_mask.expand(B, -1)
        logits, _ = self.model.class_predictor(feats, q, qm)
        pred = self.model.box_predictor(feats, feature_map) if boxes else None
        return logits, pred

    def verify_heads(self, pil_img):
        """Embed the queries, then check heads() against the full forward."""
        torch = rt.torch
        inputs = self.processor(text=[["x"]], images=pil_img,
                                return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device)
        self._embed_queries(pv)
        with torch.no_grad():
            ref = self.model(input_ids=self.input_ids, pixel_values=pv,
                             attention_mask=self.attention_mask)
            logits, boxes = self.heads(pv)
        dl = float((ref.logits - logits).abs().max())
        db = float((ref.pred_boxes - boxes).abs().max())
        if dl > 1e-3 or db > 1e-4:
            raise RuntimeError(
                f"cached-query heads differ from the full forward (logits "
                f"{dl:.2e}, boxes {db:.2e}); the attack would not see the "
                f"model inference uses.")
        return dl, db

    def detection_fn(self):
        """pil -> [{bbox xywh, category_id, score}], via the real processor.

        Per patch: the best query (score = its sigmoid), conf >= --conf,
        class-aware NMS at --nms-iou, at most --max-det boxes (the YOLO
        settings: conf 0.001, IoU 0.5, 300). The processor resizes the image
        to 768x768 without keeping the aspect ratio, so the normalized boxes
        scale back by the image width and height.
        """
        torch = rt.torch
        from torchvision.ops import batched_nms
        a = self.args

        def run_inference(pil_img):
            W, H = pil_img.size
            with torch.no_grad():
                pv = self.processor(images=pil_img, return_tensors="pt")[
                    "pixel_values"].to(self.device)
                logits, boxes = self.heads(pv)
                probs = torch.sigmoid(logits[0])
                scores, labels = probs.max(dim=-1)
                keep = scores >= a.conf
                scores, labels, b = scores[keep], labels[keep], boxes[0][keep]
                cx, cy, bw, bh = b.unbind(-1)
                xyxy = torch.stack([(cx - bw / 2) * W, (cy - bh / 2) * H,
                                    (cx + bw / 2) * W, (cy + bh / 2) * H], -1)
                xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, W)
                xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, H)
                k = batched_nms(xyxy, scores, labels, a.nms_iou)[:a.max_det]
            out = []
            for i in k.tolist():
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                out.append({"bbox": [x1, y1, x2 - x1, y2 - y1],
                            "category_id": self.cat_ids[int(labels[i])],
                            "score": float(scores[i])})
            return out

        return run_inference


def preprocess_gap(adapter, pil_img):
    """max |processor(pil) - the differentiable preprocessing| (normalized)."""
    torch = rt.torch
    with torch.no_grad():
        ref = adapter.processor(images=pil_img, return_tensors="pt")[
            "pixel_values"].float()
        mine = fl.preprocess_differentiable(
            rt.pc.pil_to_tensor(pil_img, "cpu"), adapter.geom)
    if tuple(ref.shape) != tuple(mine.shape):
        raise RuntimeError(f"processor gives {tuple(ref.shape)}, the "
                           f"differentiable path {tuple(mine.shape)}")
    return float((ref - mine).abs().max()), float((ref - mine).abs().mean())


# ======================================================================
# 2. Objectives: fn(adv, need_grad) -> (J, dJ/dadv or None), lower = stronger.
# ======================================================================
def patch_scores(adapter, x):
    """[B,3,H,W] in [0,1] -> [B, P] max-class confidence per patch."""
    pv = fl.preprocess_differentiable(x, adapter.geom)
    logits, _ = adapter.heads(pv, boxes=False)
    return rt.torch.sigmoid(logits).max(dim=-1).values


def oblivious_objective(adapter):
    """J of the bare model: the attacker does not know a defense exists."""
    torch = rt.torch

    def objective(adv, need_grad):
        if not need_grad:
            with torch.no_grad():
                return float(patch_scores(adapter, adv).sum()), None
        x = adv.detach().requires_grad_(True)
        J = patch_scores(adapter, x).sum()
        (g,) = torch.autograd.grad(J, x)
        return float(J.detach()), g.detach()

    return objective


def eot_objective(adapter, branches, args):
    """J through the defense branches (BPDA), aggregated by --eot-loss.

    The k branch inputs go through the model as ONE batch (OWL-ViT-B/32 is
    small). Each branch forwards the deployed transform's exact value and
    differentiates its surrogate (ya.bpda_branch); the aggregation is
    ya.union_objective (max: per-patch max over branches, the gradient is
    the winner mask ya.union_weights; mean: the mean of the branch J's).
    """
    torch = rt.torch
    k = len(branches)

    def objective(adv, need_grad):
        adv = adv.detach()
        values = ya.deployed_values(adv, branches, args)
        if not need_grad:
            with torch.no_grad():
                S = patch_scores(adapter, torch.cat(values, 0)).view(k, 1, -1)
            return float(ya.union_objective(S, args.eot_loss)), None
        x = adv.clone().requires_grad_(True)
        X = torch.cat([ya.bpda_branch(x, b, args, values[i])
                       for i, b in enumerate(branches)], 0)
        S = patch_scores(adapter, X).view(k, 1, -1)
        if args.eot_loss == "max" and k > 1:
            J = (S * ya.union_weights(S.detach())).sum()
        else:
            J = S.sum(dim=(1, 2)).mean()
        (g,) = torch.autograd.grad(J, x)
        return float(ya.union_objective(S.detach(), args.eot_loss)), g.detach()

    return objective


def attack_meta_note(args):
    return (f"PGD eps={args.eps} in pixel space at native resolution (uint8 "
            f"output clipped to +/- {ya.uint8_budget(args.eps)}/255), "
            f"{args.pgd_iters} steps, alpha={args.alpha}, seed {args.seed}, "
            f"{args.restarts} restart(s), iterate selection {args.pgd_select}")


# ======================================================================
# 3. Experiment: the oblivious survey (every transform, clean + attacked).
# ======================================================================
def oblivious_survey(adapter, args, coco_gt, files, image_ids):
    torch = rt.torch
    device = adapter.device
    ensembles = core.load_ensembles()
    base = core.branch_names(ensembles)
    names = base + list(SURVEY_EXTRA)
    infer = adapter.detection_fn()
    probe = rt.Image.open(os.path.join(args.image_dir, files[0])).convert("RGB")
    gap_max, gap_mean = preprocess_gap(adapter, probe)
    print(f"[guard] differentiable preprocessing vs the processor on "
          f"{files[0]}: max |diff| {gap_max:.4f}, mean {gap_mean:.5f} "
          f"(normalized units; 1/255 is about 0.015)")

    conditions = (["clean"] + [f"clean+{n}" for n in names]
                  + [OBL] + [f"{OBL}+{n}" for n in names])
    sig = rt.pc.config_signature(
        phase=4, experiment="oblivious_survey", model=adapter.model_name,
        objective="sum over patches of the max-class sigmoid, native pixels",
        branches=sorted(names), ensembles=sorted(ensembles), eps=args.eps,
        iters=args.pgd_iters, alpha=args.alpha, seed=args.seed,
        init=ya.PGD_INIT_SCHEME, restarts=args.restarts,
        quantize=args.quantize, pgd_select=args.pgd_select,
        jpeg_quality=args.jpeg_quality, median_kernel=args.median_kernel,
        tvm_weight=args.tvm_weight, tvm_iters=args.tvm_iters,
        gaussian_sigma=args.gaussian_sigma, svd_keep=args.svd_keep_ratio,
        nms_iou=args.nms_iou, n=len(files), ids=core.ids_digest(image_ids),
        device=str(device), **adapter.sig_fields(args))
    ckpt = core.make_checkpoint(args, "survey", sig)
    rt.pc.print_banner(
        f"OBLIVIOUS SURVEY (OWL-ViT) | n={len(files)} | {len(names)} "
        f"transforms + {len(ensembles)} ensembles | PGD eps={args.eps} "
        f"x{args.pgd_iters}", width=80)

    def defend(pil_img, tag, buckets, img_id):
        out = core.build_branches(pil_img, device, args, base)
        out.update(core.build_extra_branches(pil_img, SURVEY_EXTRA,
                                             seed=img_id))
        for n in names:
            core.record_dets(buckets, f"{tag}+{n}", infer(out[n]), img_id)

    def compute_one(pil_img, img_id):
        buckets = {}
        core.record_dets(buckets, "clean", infer(pil_img), img_id)
        defend(pil_img, "clean", buckets, img_id)
        frame = fl.native_frame(pil_img, device)
        obj = oblivious_objective(adapter)
        adv, info = ya.run_pgd(frame, obj, args, img_id)
        core.record_dets(buckets, OBL, infer(adv), img_id)
        defend(adv, OBL, buckets, img_id)
        buckets[META] = {"bare_objective": {
            "clean": obj(frame.x0, need_grad=False)[0],
            "oblivious": info["objective"]}}
        return buckets

    elapsed, computed, failures = core.run_per_image(
        args, "survey", files, conditions, compute_one, ckpt,
        log_every=min(getattr(args, "log_every", 100), 25))
    ckpt.flush()
    _, cond = core.score_detection(args, ckpt, names, ensembles, coco_gt,
                                   attack_tags=[OBL])
    clean, atk = cond["clean"]["mAP"], cond[OBL]["mAP"]
    damage = clean - atk
    table = {}
    for n in names + list(ensembles):
        c, a = cond[f"clean+{n}"]["mAP"], cond[f"{OBL}+{n}"]["mAP"]
        table[n] = {"label": core.paper_label(n),
                    "kind": "ensemble" if n in ensembles else "solo",
                    "clean_mAP": c, "clean_cost": c - clean,
                    "attacked_mAP": a, "recovery": a - atk,
                    "recovery_pct_of_damage": (100.0 * (a - atk) / damage
                                               if damage > 0 else None)}
    order = sorted(table, key=lambda n: -table[n]["attacked_mAP"])
    best_solo = max((n for n in table if table[n]["kind"] == "solo"),
                    key=lambda n: table[n]["attacked_mAP"])
    best_ens = max(ensembles, key=lambda n: table[n]["attacked_mAP"])
    head = {"clean_mAP": clean, "attacked_mAP": atk, "attack_damage": damage,
            "best_solo": best_solo,
            "best_solo_attacked_mAP": table[best_solo]["attacked_mAP"],
            "best_ensemble": best_ens,
            "best_ensemble_attacked_mAP": table[best_ens]["attacked_mAP"],
            "ensembles_above_best_solo": [
                n for n in ensembles
                if table[n]["attacked_mAP"] > table[best_solo]["attacked_mAP"]]}

    payload = {
        "phase": 4, "experiment": "oblivious_survey",
        "model": adapter.model_name, "model_key": args.model_key,
        "smoke_test": bool(args.smoke_test),
        "attack": {"type": "PGD (oblivious: the bare model)", "eps": args.eps,
                   "iters": args.pgd_iters, "alpha": args.alpha,
                   "seed": args.seed, "restarts": args.restarts,
                   "quantize": args.quantize,
                   "iterate_selection": args.pgd_select,
                   "init_scheme": ya.PGD_INIT_SCHEME,
                   "pixel_space": "raw [0,1], native resolution",
                   "objective": "minimize the sum over patches of the "
                                "max-class confidence"},
        "preprocess_gap_normalized": {"max": gap_max, "mean": gap_mean},
        "execution": {"deterministic": bool(getattr(args, "deterministic",
                                                    False)),
                      "log_name": args.log_name},
        "config": {k: getattr(args, k) for k in
                   ("jpeg_quality", "median_kernel", "tvm_weight",
                    "tvm_iters", "gaussian_sigma", "svd_keep_ratio",
                    "nms_iou")},
        "model_config": adapter.sig_fields(args),
        "branches": names, "ensembles": ensembles,
        "ensemble_paper_names": core.ENSEMBLE_PAPER_NAMES,
        "num_images": len(ckpt.buckets), "images_computed_this_run": computed,
        "select": args.select, "seed": args.seed, "image_ids": image_ids,
        "image_ids_sha1_16": core.ids_digest(image_ids),
        "device": str(device), "failures": failures, "seconds": elapsed,
        "seconds_per_image": elapsed / computed if computed else None,
        "headline": head, "defenses": table, "conditions": cond,
    }
    md = [f"# Oblivious survey: every transform on {adapter.model_name}\n"]
    md += core.header_lines(args, image_ids, device=device, extra=[
        f"- {attack_meta_note(args)}; the attacker ignores the defense",
        "- COCO mAP@[.5:.95] with OWL-ViT's own confidences"])
    md.append(f"Clean **{clean:.4f}**, attacked undefended **{atk:.4f}** "
              f"(damage {damage:.4f}). Best single transform under attack: "
              f"{core.paper_label(best_solo)} "
              f"{table[best_solo]['attacked_mAP']:.4f}; best ensemble: "
              f"{core.paper_label(best_ens)} "
              f"{table[best_ens]['attacked_mAP']:.4f}.\n")
    md.append("| Defense | Kind | Clean mAP | Clean cost | Attacked mAP | "
              "Recovery | % of damage |")
    md.append("|---|---|---|---|---|---|---|")
    for n in order:
        t = table[n]
        md.append(f"| {t['label']} | {t['kind']} | {t['clean_mAP']:.4f} | "
                  f"{t['clean_cost']:+.4f} | {t['attacked_mAP']:.4f} | "
                  f"{t['recovery']:+.4f} | "
                  f"{core.fmt(t['recovery_pct_of_damage'], 1)} |")
    md.append("")
    md += core.timing_lines(elapsed, computed, len(files), str(device),
                            targets=(300, 1000, 5000))
    stem = f"oblivious_survey_{args.model_key}_seed{args.seed}"
    core.write_per_image(args, stem, {
        "note": "per-image detection counts and the bare objective",
        "counts": {str(i): {c: len(b.get(c, [])) for c in conditions}
                   for i, b in ckpt.buckets.items()},
        "attack_meta": {str(i): b.get(META) for i, b in ckpt.buckets.items()}})
    core.write_summary(args, payload, stem)
    core.write_markdown(args, "\n".join(md), stem)
    print("\n" + "=" * 70)
    print("RESULT DIGEST: oblivious survey (full table in the .md)")
    print("=" * 70)
    print(f"  clean {clean:.4f} | attacked {atk:.4f}")
    for n in order[:5]:
        print(f"  {table[n]['label']:<24} attacked {table[n]['attacked_mAP']:.4f}"
              f"  clean {table[n]['clean_mAP']:.4f}")
    core.print_runtime_estimate(elapsed, computed, str(device))
    return payload


# ======================================================================
# 4. Experiment: the adaptive attack (BPDA + EOT through --target).
# ======================================================================
def adaptive_signature(args, adapter, target, branches, files, image_ids):
    return rt.pc.config_signature(
        phase=4, experiment="adaptive_bpda", model=adapter.model_name,
        objective="sum over patches of the max-class sigmoid, native pixels",
        target=target, branches=sorted(branches), eps=args.eps,
        iters=args.pgd_iters, alpha=args.alpha, seed=args.seed,
        init=ya.PGD_INIT_SCHEME, surrogates=ya.SURROGATE_VERSION,
        restarts=args.restarts, eot_loss=args.eot_loss,
        median_surrogate=args.median_surrogate, quantize=args.quantize,
        attack_space=args.attack_space, pgd_select=args.pgd_select,
        jpeg_quality=args.jpeg_quality, median_kernel=args.median_kernel,
        gaussian_sigma=args.gaussian_sigma, tvm_weight=args.tvm_weight,
        tvm_iters=args.tvm_iters, nms_iou=args.nms_iou, n=len(files),
        ids=core.ids_digest(image_ids), device=str(adapter.device),
        **adapter.sig_fields(args))


def adaptive_bpda(adapter, args, coco_gt, files, image_ids):
    target = args.target
    ensembles, branches, paper = ya.resolve_target(target)
    is_ens = bool(ensembles)
    device = adapter.device
    infer = adapter.detection_fn()
    ya.check_surrogates(device, args, branches)
    probe = rt.Image.open(os.path.join(args.image_dir, files[0])).convert("RGB")
    gap_max, gap_mean = preprocess_gap(adapter, probe)
    print(f"[guard] differentiable preprocessing vs the processor on "
          f"{files[0]}: max |diff| {gap_max:.4f}, mean {gap_mean:.5f}")

    conditions = (["clean"] + [f"clean+{b}" for b in branches]
                  + [OBL] + [f"{OBL}+{b}" for b in branches]
                  + [ADP] + [f"{ADP}+{b}" for b in branches])
    ckpt = core.make_checkpoint(args, "adaptive", adaptive_signature(
        args, adapter, target, branches, files, image_ids))
    rt.pc.print_banner(
        f"ADAPTIVE WHITE-BOX (OWL-ViT): BPDA vs {paper} | n={len(files)} | "
        f"PGD eps={args.eps} x{args.pgd_iters} | eot={args.eot_loss}",
        width=80)

    def defended(pil_img, tag, buckets, img_id):
        for b, img in core.build_branches(pil_img, device, args,
                                          branches).items():
            core.record_dets(buckets, f"{tag}+{b}", infer(img), img_id)

    def compute_one(pil_img, img_id):
        buckets = {}
        core.record_dets(buckets, "clean", infer(pil_img), img_id)
        defended(pil_img, "clean", buckets, img_id)
        frame = fl.native_frame(pil_img, device)
        through = eot_objective(adapter, branches, args)
        adv_obl, obl = ya.run_pgd(frame, oblivious_objective(adapter), args,
                                  img_id)
        core.record_dets(buckets, OBL, infer(adv_obl), img_id)
        defended(adv_obl, OBL, buckets, img_id)
        adv_adp, adp = ya.run_pgd(frame, through, args, img_id)
        core.record_dets(buckets, ADP, infer(adv_adp), img_id)
        defended(adv_adp, ADP, buckets, img_id)
        buckets[META] = {
            "target_objective": {
                "clean": through(frame.x0, need_grad=False)[0],
                "oblivious": through(obl["adv_q"], need_grad=False)[0],
                "adaptive": adp["objective"]},
            "oblivious_bare_objective": obl["objective"]}
        return buckets

    elapsed, computed, failures = core.run_per_image(
        args, "bpda", files, conditions, compute_one, ckpt,
        log_every=min(getattr(args, "log_every", 100), 25))
    ckpt.flush()
    _, cond = core.score_detection(args, ckpt, branches, ensembles, coco_gt,
                                   attack_tags=[OBL, ADP])
    head = {
        "clean_mAP": cond["clean"]["mAP"],
        "clean_defended_mAP": cond[f"clean+{target}"]["mAP"],
        "oblivious_attacked_undefended_mAP": cond[OBL]["mAP"],
        "oblivious_attacked_defended_mAP": cond[f"{OBL}+{target}"]["mAP"],
        "adaptive_attacked_undefended_mAP": cond[ADP]["mAP"],
        "adaptive_attacked_defended_mAP": cond[f"{ADP}+{target}"]["mAP"],
    }
    metas = [b[META] for b in ckpt.buckets.values() if META in b]
    tj = {k: [m["target_objective"][k] for m in metas]
          for k in ("clean", "oblivious", "adaptive")}
    below = [a < o for a, o in zip(tj["adaptive"], tj["oblivious"])]
    mean = lambda xs: sum(xs) / len(xs) if xs else None      # noqa: E731
    surrogate = {
        "definition": (f"J = sum over patches of the max-class confidence, "
                       f"measured THROUGH {paper} (eot-loss {args.eot_loss}); "
                       f"lower = more suppression."),
        "images": len(metas), "mean_clean": mean(tj["clean"]),
        "mean_oblivious_adv": mean(tj["oblivious"]),
        "mean_adaptive_adv": mean(tj["adaptive"]),
        "frac_images_adaptive_below_oblivious":
            (sum(below) / len(below)) if below else None}
    surrogates = {b: ya.surrogate_description(b, args) for b in branches}
    payload = {
        "phase": 4, "experiment": "adaptive_bpda",
        "model": adapter.model_name, "model_key": args.model_key,
        "smoke_test": bool(args.smoke_test), "target": target,
        "attack_flags": {k: getattr(args, k) for k in ya.ATTACK_FLAGS},
        "defense": {"name": paper, "key": target,
                    "kind": "ensemble" if is_ens else "solo",
                    "branches": branches, "nms_iou": args.nms_iou},
        "attack": {"type": "PGD", "eps": args.eps, "iters": args.pgd_iters,
                   "alpha": args.alpha, "seed": args.seed,
                   "restarts": args.restarts, "quantize": args.quantize,
                   "iterate_selection": args.pgd_select,
                   "init_scheme": ya.PGD_INIT_SCHEME,
                   "pixel_space": "raw [0,1], native resolution",
                   "objective": "minimize the sum over patches of the "
                                "max-class confidence"},
        "bpda": {"surrogates": surrogates, "eot_loss": args.eot_loss},
        "preprocess_gap_normalized": {"max": gap_max, "mean": gap_mean},
        "execution": {"deterministic": bool(getattr(args, "deterministic",
                                                    False)),
                      "log_name": args.log_name},
        "config": {k: getattr(args, k) for k in
                   ("jpeg_quality", "median_kernel", "tvm_weight",
                    "tvm_iters", "gaussian_sigma", "nms_iou")},
        "model_config": adapter.sig_fields(args),
        "num_images": len(ckpt.buckets), "images_computed_this_run": computed,
        "select": args.select, "seed": args.seed, "image_ids": image_ids,
        "image_ids_sha1_16": core.ids_digest(image_ids),
        "device": str(device), "failures": failures, "seconds": elapsed,
        "seconds_per_image": elapsed / computed if computed else None,
        "headline": head, "surrogate_objective": surrogate,
        "conditions": cond,
    }
    md = [f"# Adaptive white-box: BPDA vs {paper} ({adapter.model_name})\n"]
    md += core.header_lines(args, image_ids, device=device, extra=[
        f"- Target **{paper}** (`{target}`) = {' + '.join(branches)}",
        f"- {attack_meta_note(args)}; ensemble objective `{args.eot_loss}`"])
    md.append("| Condition | mAP |")
    md.append("|---|---|")
    for key, val in head.items():
        md.append(f"| {key} | {val:.4f} |")
    md.append("")
    md.append(f"- Images where the adaptive J is below the oblivious J: "
              f"**{core.fmt(surrogate['frac_images_adaptive_below_oblivious'], 3)}**"
              f" of {surrogate['images']}")
    md.append("- BPDA: " + "; ".join(f"{b}: {d}" for b, d in surrogates.items()))
    md.append("")
    md += core.timing_lines(elapsed, computed, len(files), str(device),
                            targets=(300, 1000, 5000))
    stem = ya.summary_stem(args, STEM_FLAG_DEFAULTS)
    core.write_per_image(args, stem, {
        "note": "per-image detection counts and the attacker's objectives",
        "counts": {str(i): {c: len(b.get(c, [])) for c in conditions}
                   for i, b in ckpt.buckets.items()},
        "attack_meta": {str(i): b.get(META) for i, b in ckpt.buckets.items()}})
    core.write_summary(args, payload, stem)
    core.write_markdown(args, "\n".join(md), stem)
    print("\n" + "=" * 70)
    print(f"RESULT DIGEST: adaptive vs {paper} (full table in the .md)")
    print("=" * 70)
    for key, val in head.items():
        print(f"  {key:<38}: {val:+.4f}")
    print(f"  adaptive J below oblivious J on "
          f"{core.fmt(surrogate['frac_images_adaptive_below_oblivious'], 3)}"
          f" of images")
    core.print_runtime_estimate(elapsed, computed, str(device))
    return payload


# ======================================================================
# 5. Entry point.
# ======================================================================
def parse_args(argv=None):
    p = core.base_parser("Phase-4 OWL-ViT runs (fourth model).", EXPERIMENTS)
    ya.add_attack_arguments(p, native_only=True)
    m = p.add_argument_group("model")
    m.add_argument("--owl-model", default="google/owlvit-base-patch32")
    m.add_argument("--prompt", default="a photo of a {}",
                   help="Text query per COCO class ({} = the class name).")
    m.add_argument("--conf", type=float, default=0.001,
                   help="Score threshold before NMS (YOLO's value).")
    m.add_argument("--max-det", type=int, default=300,
                   help="Boxes kept per image after NMS (YOLO's value).")
    args = p.parse_args(argv)
    core.finalize_args(args, TRACK)
    ya.finalize_attack_args(args)
    return args


def preflight_weights(args):
    hub, how = fl.hf_hub_cache()
    present, repo, detail = fl.hf_cached(hub, args.owl_model, "main")
    offline = (f"HF_HUB_OFFLINE is set and {detail}" if fl.hf_offline()
               else None)
    return [(args.owl_model, present, f"{repo} ({how}; {detail})", offline)]


def main(argv=None):
    args = parse_args(argv)
    if args.preflight:
        return core.preflight(
            args,
            modules=[("numpy", "all"), ("PIL", "all"), ("torch", "all"),
                     ("torchvision", "NMS"), ("transformers", "OWL-ViT"),
                     ("pycocotools", "COCO mAP")],
            weights=preflight_weights(args))

    device = core.load_runtime(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with core.run_log(core.log_path(args), argv=sys.argv,
                      extra={"experiment": args.experiment, "n": args.n,
                             "device": str(device),
                             "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                             "smoke_test": args.smoke_test}):
        args.image_dir = core.resolve_image_dir(args.image_dir)
        args.ann_file = core.resolve_ann_file(args.ann_file)
        files, image_ids = core.select_images(args.image_dir, args.n, args.seed,
                                             args.select,
                                             allow_short=args.smoke_test)
        args.n = len(files)
        rt.pc.print_banner(f"PHASE 4 — OWL-ViT — {args.experiment}", width=80)
        if args.smoke_test:
            print(f"*** SMOKE TEST: {args.n} images on {device}. The numbers "
                  f"below are NOT results. ***")
        print(f"  run id    : {args.run_id}")
        print(f"  images    : {args.n} from {args.image_dir} "
              f"(--select {args.select}, seed {args.seed})")
        print(f"  id digest : {core.ids_digest(image_ids)}")
        print(f"  device    : {device}")
        print(f"  output    : {args.output_dir}")
        print(f"  log       : {args.log_name}")

        from pycocotools.coco import COCO
        coco_gt = COCO(args.ann_file)
        adapter = OwlAdapter(args, device)
        adapter.load(coco_gt)
        probe = rt.Image.open(os.path.join(args.image_dir, files[0])
                              ).convert("RGB")
        dl, db = adapter.verify_heads(probe)
        print(f"[guard] cached-query heads vs the full forward: logits "
              f"{dl:.2e}, boxes {db:.2e}")

        if args.experiment == "oblivious_survey":
            oblivious_survey(adapter, args, coco_gt, files, image_ids)
        elif args.experiment == "adaptive_bpda":
            adaptive_bpda(adapter, args, coco_gt, files, image_ids)
        else:  # pragma: no cover — argparse restricts the choices
            raise ValueError(f"unknown experiment {args.experiment}")
        core.finish(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
