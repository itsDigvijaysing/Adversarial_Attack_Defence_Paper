#!/usr/bin/env python3
# ======================================================================
# p4_smolvlm.py: a third, chat-style model (SmolVLM-256M-Instruct, 0.26 B
# parameters) for the "does the defense carry over without retraining"
# question.
#
#   python phase4/p4_smolvlm.py --n 1000 --gpu 0
#   python phase4/p4_smolvlm.py --smoke-test --device cuda --gpu 0
#   python phase4/p4_smolvlm.py --preflight
#
# THE MODEL WAS CHOSEN BEFORE ANY RUN (17 Nov 2026) and is reported whatever
# it shows. A chat model returns text, not boxes, so the task is different
# from the two detection tracks:
#
# TASK (experiment `presence_qa`). For each image, --questions-per-image
# yes/no questions "Is there a <class> in this image?", half about a COCO
# class that IS annotated in the image and half about one that is not
# (seeded per image). The score is ACCURACY against the COCO annotations: a
# ground-truth metric, unlike the OCR self-consistency number. The answer is
# read from the next-token logits (yes-tokens against no-tokens), so there is
# no free-form generation and no parsing.
#
# ATTACK. Oblivious PGD only (the attacker ignores the defense): one
# perturbation per image, raw [0,1] pixel space, eps=0.03 (7/255 after the
# uint8 clip), maximizing the cross-entropy of the CORRECT answers of that
# image's questions. No adaptive attack is implemented for this model.
#
# DEFENSES. Every solo transform (the 7 Phase-4 branches plus the other
# survey transforms) and the 7 ensembles, all from phase3_common. An ensemble
# answers with the MEAN of its branches' yes-probabilities (there are no
# boxes to merge).
#
# RESOLUTION. Every image is first resized to a --side square (default 512,
# the model's native tile), in every condition, and image splitting is off.
# The processor is then an identity resize plus normalization, so the
# attacker's preprocessing is exact (checked at startup) and the saved
# adversarial image is exactly what the model and the defenses receive.
# Transforms therefore run on the 512 square, not on the original image.
# ======================================================================

from __future__ import annotations

import os
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p4_core as core                                     # noqa: E402
import p4_yolo as ya              # noqa: E402  (PGD loop, seeded init only)
from p4_core import rt                                     # noqa: E402

EXPERIMENTS = ("presence_qa",)
DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
ATTACK_TAG = "pgd"
Q_KEY = "_questions"
YES_WORDS = ("Yes", " Yes", "yes", " yes")
NO_WORDS = ("No", " No", "no", " no")
SLOW_BRANCHES = ("nlm", "bm3d")


def question_text(class_name):
    return f"Is there a {class_name} in this image? Answer yes or no."


def pick_questions(present_ids, all_ids, k, seed, image_id):
    """[(category_id, label)] for one image: label 1 = present, 0 = absent.

    Half present and half absent when the image has enough annotated
    classes; an image with no annotations gets k absent questions. Seeded by
    (seed, image id) with the stdlib, so it is the same on every machine.
    """
    rng = random.Random(f"presence|{int(seed)}|{int(image_id)}")
    present = sorted(set(present_ids))
    absent = sorted(set(all_ids) - set(present))
    n_pos = min(k // 2, len(present))
    n_neg = min(k - n_pos, len(absent))
    return ([(c, 1) for c in rng.sample(present, n_pos)]
            + [(c, 0) for c in rng.sample(absent, n_neg)])


def ensemble_probs(per_branch, members):
    """Mean yes-probability over the member branches, per question."""
    cols = [per_branch[m] for m in members]
    return [sum(v) / len(v) for v in zip(*cols)]


def accuracy_table(buckets, tags):
    """{tag: (accuracy, {image_id: number correct})} over all questions."""
    out = {}
    for tag in tags:
        correct, total, per_img = 0, 0, {}
        for iid, b in buckets.items():
            labels = [q[1] for q in b[Q_KEY]]
            hits = sum(int((p > 0.5) == bool(y))
                       for p, y in zip(b[tag], labels))
            per_img[iid] = hits
            correct += hits
            total += len(labels)
        out[tag] = (correct / total if total else float("nan"), per_img)
    return out


def paired_ci(per_img_a, per_img_b, n_questions, B=1000, seed=42):
    """95% percentile interval of accuracy(a) - accuracy(b), resampling images."""
    np = rt.np
    ids = sorted(per_img_a)
    diff = np.array([per_img_a[i] - per_img_b[i] for i in ids], dtype=float)
    cnt = np.array([n_questions[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    boots = diff[idx].sum(axis=1) / np.maximum(cnt[idx].sum(axis=1), 1.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": float(diff.sum() / max(cnt.sum(), 1.0)),
            "lo": float(lo), "hi": float(hi),
            "excludes_0": bool(lo > 0 or hi < 0)}


class SmolAdapter:
    """Loads the chat model once; yes-probabilities and the attack loss."""

    def __init__(self, args, device):
        self.args, self.device = args, device
        self.model_name = args.model
        self.model = self.processor = None
        self.yes_ids = self.no_ids = None
        self.mean = self.std = None

    def sig_fields(self, args):
        return {"vlm": args.model, "revision": args.revision,
                "side": args.side, "questions": args.questions_per_image,
                "prompt": question_text("{class}")}

    def load(self):
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as Auto
        except ImportError:                       # older transformers
            from transformers import AutoModelForVision2Seq as Auto
        a, torch = self.args, rt.torch
        kw = {"revision": a.revision} if a.revision else {}
        self.processor = AutoProcessor.from_pretrained(
            a.model, size={"longest_edge": a.side}, do_image_splitting=False,
            **kw)
        self.model = Auto.from_pretrained(
            a.model, torch_dtype=torch.float32, **kw).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        tok = self.processor.tokenizer

        def first_ids(words):
            return sorted({tok.encode(w, add_special_tokens=False)[0]
                           for w in words})

        self.yes_ids, self.no_ids = first_ids(YES_WORDS), first_ids(NO_WORDS)
        assert not set(self.yes_ids) & set(self.no_ids), (
            f"yes and no share a first token: {self.yes_ids} {self.no_ids}")
        ip = self.processor.image_processor
        self.mean = [float(v) for v in ip.image_mean]
        self.std = [float(v) for v in ip.image_std]
        print(f"[smolvlm] loaded {a.model} | side={a.side} | yes ids "
              f"{self.yes_ids} | no ids {self.no_ids} | device={self.device}")

    def encode(self, pils, question):
        """Processor output for N images with the same question, on device."""
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True)
        enc = self.processor(text=[prompt] * len(pils),
                             images=[[p] for p in pils],
                             return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in enc.items()}

    def margin(self, logits):
        """log-odds of yes against no from next-token logits [N, vocab]."""
        torch = rt.torch
        return (torch.logsumexp(logits[:, self.yes_ids], dim=1)
                - torch.logsumexp(logits[:, self.no_ids], dim=1))

    def yes_probs(self, pils, question):
        """P(yes) for each image: scored inference, through the processor."""
        torch = rt.torch
        with torch.no_grad():
            enc = self.encode(pils, question)
            logits = self.model(**enc).logits[:, -1, :].float()
            return torch.sigmoid(self.margin(logits)).tolist()

    def normalize(self, x, like):
        """[1,3,S,S] in [0,1] -> the processor's pixel_values layout."""
        torch = rt.torch
        m = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        s = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        return ((x - m) / s).view(like.shape)

    def check_preprocessing(self, pil_sq, tol=0.02):
        """Refuse to run unless the processor is resize-free on the square."""
        enc = self.encode([pil_sq], question_text("person"))
        pv = enc["pixel_values"]
        s = self.args.side
        assert tuple(pv.shape[-3:]) == (3, s, s) and pv.numel() == 3 * s * s, (
            f"the processor returns pixel_values {tuple(pv.shape)} for a "
            f"{s}x{s} image; expected one {s}x{s} tile. Image splitting or a "
            f"resize is still active, so the attack would not reach the "
            f"model the way inference does. Refusing to run.")
        x = rt.pc.pil_to_tensor(pil_sq, self.device)
        gap = float((pv.float() - self.normalize(x, pv)).abs().max())
        assert gap < tol, (
            f"processor pixel_values differ from (x - mean) / std by "
            f"{gap:.4f} (> {tol}); the processor is not an identity resize "
            f"on the square. Refusing to run.")
        print(f"[guard] processor == normalize on the {s}x{s} square "
              f"(max |diff| = {gap:.2e})")
        return gap

    def attack_objective(self, pil_sq, questions, names):
        """J(adv) = sum over questions of log P(correct answer); PGD lowers it."""
        torch, F = rt.torch, rt.torch.nn.functional
        encs = [self.encode([pil_sq], question_text(names[c]))
                for c, _ in questions]
        signs = [1.0 if y else -1.0 for _, y in questions]

        def total(x):
            j = 0.0
            for enc, sign in zip(encs, signs):
                feed = dict(enc)
                feed["pixel_values"] = self.normalize(x, enc["pixel_values"])
                logits = self.model(**feed).logits[:, -1, :].float()
                j = j + F.logsigmoid(sign * self.margin(logits)).sum()
            return j

        def objective(adv, need_grad):
            if not need_grad:
                with torch.no_grad():
                    return float(total(adv)), None
            x = adv.detach().requires_grad_(True)
            j = total(x)
            (g,) = torch.autograd.grad(j, x)
            return float(j.detach()), torch.nan_to_num(g)

        return objective


def branch_list(args):
    """Every solo transform scored here (base 7 + the other survey ones)."""
    import importlib.util
    names = list(core.SOLO_PAPER_NAMES) + list(core.EXTRA_PAPER_NAMES)
    if args.skip_slow:
        names = [n for n in names if n not in SLOW_BRANCHES]
    if "bm3d" in names and importlib.util.find_spec("bm3d") is None:
        print("[presence_qa] bm3d is not installed: dropping the BM3D row.")
        names.remove("bm3d")
    return names


def presence_qa(adapter, args, coco_gt, files, image_ids):
    device = adapter.device
    ensembles = core.load_ensembles()
    names = branch_list(args)
    base = [n for n in names if n in core.SOLO_PAPER_NAMES]
    extra = [n for n in names if n not in core.SOLO_PAPER_NAMES]
    cats = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    side = args.side

    def square(pil_img):
        return pil_img.resize((side, side), rt.Image.BICUBIC)

    probe = square(rt.Image.open(
        os.path.join(args.image_dir, files[0])).convert("RGB"))
    gap = adapter.check_preprocessing(probe)

    def defend(pil_sq, img_id):
        out = core.build_branches(pil_sq, device, args, base)
        out.update(core.build_extra_branches(pil_sq, extra, seed=img_id))
        return out

    solo_tags = (["clean"] + [f"clean+{n}" for n in names]
                 + [ATTACK_TAG] + [f"{ATTACK_TAG}+{n}" for n in names])
    sig = rt.pc.config_signature(
        phase=4, experiment="presence_qa", branches=sorted(names),
        eps=args.eps, iters=args.pgd_iters, alpha=args.alpha, seed=args.seed,
        init=ya.PGD_INIT_SCHEME, quantize=args.quantize,
        pgd_select=args.pgd_select, restarts=args.restarts,
        jpeg_quality=args.jpeg_quality, median_kernel=args.median_kernel,
        tvm_weight=args.tvm_weight, tvm_iters=args.tvm_iters,
        gaussian_sigma=args.gaussian_sigma, svd_keep=args.svd_keep_ratio,
        n=len(files), ids=core.ids_digest(image_ids), device=str(device),
        **adapter.sig_fields(args))
    ckpt = core.make_checkpoint(args, "presence", sig)

    rt.pc.print_banner(
        f"PRESENCE QA: {adapter.model_name} | n={len(files)} | "
        f"{args.questions_per_image} questions/image | {len(names)} "
        f"transforms + {len(ensembles)} ensembles | oblivious PGD "
        f"eps={args.eps} x{args.pgd_iters}", width=80)

    def compute_one(pil_img, img_id):
        sq = square(pil_img)
        present = [a["category_id"] for a in coco_gt.loadAnns(
            coco_gt.getAnnIds(imgIds=[img_id]))]
        qs = pick_questions(present, list(cats), args.questions_per_image,
                            args.seed, img_id)
        frame = SimpleNamespace(space="native",
                                x0=rt.pc.pil_to_tensor(sq, device),
                                to_model=lambda t: t,
                                to_pil=lambda u8: rt.Image.fromarray(u8))
        adv, info = ya.run_pgd(frame, adapter.attack_objective(sq, qs, cats),
                               args, img_id)
        clean_def, adv_def = defend(sq, img_id), defend(adv, img_id)
        images = ([sq] + [clean_def[n] for n in names]
                  + [adv] + [adv_def[n] for n in names])
        bucket = {t: [] for t in solo_tags}
        for cat_id, _ in qs:
            probs = adapter.yes_probs(images, question_text(cats[cat_id]))
            for tag, p in zip(solo_tags, probs):
                bucket[tag].append(float(p))
        bucket[Q_KEY] = [[int(c), int(y)] for c, y in qs]
        bucket["_objective"] = info["objective"]
        return bucket

    elapsed, computed, failures = core.run_per_image(
        args, "presence", files, solo_tags, compute_one, ckpt,
        log_every=min(getattr(args, "log_every", 100), 50))
    ckpt.flush()

    # Ensembles are built from the stored branch answers: no extra model run.
    all_tags = list(solo_tags)
    for b in ckpt.buckets.values():
        for prefix in ("clean", ATTACK_TAG):
            per_branch = {n: b[f"{prefix}+{n}"] for n in names}
            for ens, members in ensembles.items():
                b[f"{prefix}+{ens}"] = ensemble_probs(per_branch, members)
    all_tags += [f"{p}+{e}" for p in ("clean", ATTACK_TAG) for e in ensembles]
    acc = accuracy_table(ckpt.buckets, all_tags)
    n_q = {iid: len(b[Q_KEY]) for iid, b in ckpt.buckets.items()}
    clean_acc, atk_acc = acc["clean"][0], acc[ATTACK_TAG][0]
    damage = clean_acc - atk_acc

    defenses = {}
    for name in names + list(ensembles):
        c, a = acc[f"clean+{name}"], acc[f"{ATTACK_TAG}+{name}"]
        rec = paired_ci(a[1], acc[ATTACK_TAG][1], n_q)
        defenses[name] = {
            "label": core.paper_label(name),
            "kind": "ensemble" if name in ensembles else "solo",
            "clean_accuracy": c[0], "attacked_accuracy": a[0],
            "clean_cost": paired_ci(c[1], acc["clean"][1], n_q),
            "recovery": rec,
            "recovery_pct_of_damage":
                (100.0 * rec["point"] / damage) if damage > 0 else None,
        }
    best = max(defenses, key=lambda k: defenses[k]["attacked_accuracy"])
    total_q = sum(n_q.values())
    labels = [q[1] for b in ckpt.buckets.values() for q in b[Q_KEY]]
    payload = {
        "phase": 4, "experiment": "presence_qa", "model": adapter.model_name,
        "model_key": args.model_key, "smoke_test": bool(args.smoke_test),
        "task": ("yes/no object-presence questions built from the COCO "
                 "instance annotations; accuracy against ground truth"),
        "prompt": question_text("{class}"),
        "answer_rule": "P(yes) > 0.5 from next-token yes/no logits",
        "ensemble_rule": "mean of the member branches' yes-probabilities",
        "attack": {"type": "PGD (oblivious only)", "eps": args.eps,
                   "iters": args.pgd_iters, "alpha": args.alpha,
                   "seed": args.seed,
                   "pixel_space": f"raw [0,1] on the {side}x{side} square",
                   "objective": ("maximize the cross-entropy of the correct "
                                 "answers of the image's questions")},
        "resolution_note": (f"every image is resized to {side}x{side} in "
                            f"every condition; image splitting is off"),
        "preprocess_gap": gap,
        "execution": {"deterministic": bool(args.deterministic),
                      "log_name": args.log_name},
        "num_images": len(ckpt.buckets), "num_questions": total_q,
        "share_yes_labels": (sum(labels) / len(labels)) if labels else None,
        "images_computed_this_run": computed, "failures": failures,
        "seconds": elapsed,
        "seconds_per_image": elapsed / computed if computed else None,
        "image_ids": image_ids,
        "image_ids_sha1_16": core.ids_digest(image_ids),
        "device": str(device), "model_config": adapter.sig_fields(args),
        "branches": names, "ensembles": ensembles,
        "headline": {"clean_accuracy": clean_acc,
                     "attacked_accuracy": atk_acc,
                     "attack_damage": damage, "best_defense": best,
                     "best_defense_attacked_accuracy":
                         defenses[best]["attacked_accuracy"]},
        "defenses": defenses,
    }

    def ci(d):
        return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"

    md = [f"# Object-presence questions: {adapter.model_name}\n"]
    md += core.header_lines(args, image_ids, device=device, extra=[
        f"- {total_q} yes/no questions "
        f"({args.questions_per_image} per image), accuracy against the COCO "
        f"annotations; chance is 0.5 on balanced questions",
        f"- Oblivious PGD eps={args.eps}, {args.pgd_iters} iters, "
        f"alpha={args.alpha:.5f}, on the {side}x{side} square",
        "- Brackets: 95% paired bootstrap intervals over images (B=1000)"])
    md.append(f"- **Clean accuracy {clean_acc:.4f} · attacked "
              f"{atk_acc:.4f}** (damage {damage:+.4f})\n")
    md.append("| Defense | Kind | Clean acc. | Clean cost | Attacked acc. | "
              "Recovery |")
    md.append("|---|---|---|---|---|---|")
    for name, d in defenses.items():
        md.append(f"| {d['label']} (`{name}`) | {d['kind']} | "
                  f"{d['clean_accuracy']:.4f} | {ci(d['clean_cost'])} | "
                  f"{d['attacked_accuracy']:.4f} | {ci(d['recovery'])} |")
    md.append("")
    md += core.timing_lines(elapsed, computed, len(files), str(device),
                            targets=(1000, 4000, 5000))
    stem = f"presence_qa_{args.model_key}"
    core.write_per_image(args, stem, {
        "note": "per-image yes-probabilities per condition and the questions",
        "buckets": {str(i): b for i, b in ckpt.buckets.items()}})
    core.write_summary(args, payload, stem)
    core.write_markdown(args, "\n".join(md), stem)
    print("\n" + "=" * 70)
    print("RESULT DIGEST (full table in the .md)")
    print("=" * 70)
    print(f"  clean accuracy     : {clean_acc:.4f}")
    print(f"  attacked accuracy  : {atk_acc:.4f}")
    print(f"  best defense       : {defenses[best]['label']} -> "
          f"{defenses[best]['attacked_accuracy']:.4f}")
    core.print_runtime_estimate(elapsed, computed, str(device),
                                targets=(1000, 4000, 5000))
    return payload


def parse_args(argv=None):
    p = core.base_parser("Phase-4 chat-model run (SmolVLM presence "
                         "questions).", EXPERIMENTS)
    m = p.add_argument_group("model and task")
    m.add_argument("--model", default=DEFAULT_MODEL)
    m.add_argument("--revision", default=None)
    m.add_argument("--side", type=int, default=512,
                   help="Square size every image is resized to (the model's "
                        "tile size; do not change without checking the "
                        "startup guard).")
    m.add_argument("--questions-per-image", type=int, default=2)
    m.add_argument("--skip-slow", action="store_true",
                   help="Drop non-local means and BM3D (seconds per image "
                        "each on the CPU).")
    g = p.add_argument_group("attack (oblivious PGD)")
    g.add_argument("--eps", type=float, default=0.03)
    g.add_argument("--pgd-iters", type=int, default=None,
                   help="Default 10 (2 under --smoke-test).")
    g.add_argument("--alpha", type=float, default=None,
                   help="Default eps/4.")
    args = p.parse_args(argv)
    core.finalize_args(args, "smolvlm")
    ya.finalize_attack_args(args)
    # The shared PGD loop's remaining knobs, fixed for this model.
    args.restarts, args.quantize, args.pgd_select = 1, "truncate", "last"
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.preflight:
        import p4_florence as flo
        hub, how = flo.hf_hub_cache()
        present, repo, detail = flo.hf_cached(hub, args.model,
                                              args.revision or "main")
        offline = ("HF_HUB_OFFLINE is set and the model is not cached"
                   if flo.hf_offline() else None)
        return core.preflight(
            args,
            modules=[("numpy", "all"), ("PIL", "all"), ("torch", "all"),
                     ("transformers", "SmolVLM"),
                     ("pycocotools", "COCO annotations")],
            weights=[(args.model, present, f"{repo} ({how}; {detail})",
                      offline)])

    device = core.load_runtime(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with core.run_log(core.log_path(args), argv=sys.argv,
                      extra={"experiment": args.experiment, "n": args.n,
                             "device": str(device),
                             "smoke_test": args.smoke_test}):
        args.image_dir = core.resolve_image_dir(args.image_dir)
        args.ann_file = core.resolve_ann_file(args.ann_file)
        files, image_ids = core.select_images(
            args.image_dir, args.n, args.seed, args.select,
            allow_short=args.smoke_test)
        args.n = len(files)
        rt.pc.print_banner(f"PHASE 4: SMOLVLM: {args.experiment}", width=80)
        if args.smoke_test:
            print(f"*** SMOKE TEST: {args.n} images on {device}. The numbers "
                  f"below are NOT results. ***")
        print(f"  run id    : {args.run_id}")
        print(f"  images    : {args.n} from {args.image_dir}")
        print(f"  device    : {device}")
        print(f"  output    : {args.output_dir}")

        from pycocotools.coco import COCO
        coco_gt = COCO(args.ann_file)
        adapter = SmolAdapter(args, device)
        adapter.load()
        presence_qa(adapter, args, coco_gt, files, image_ids)
        core.finish(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
