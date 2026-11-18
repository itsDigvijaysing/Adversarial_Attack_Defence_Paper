#!/usr/bin/env python3
"""Score the first K images of a finished adaptive_bpda checkpoint (read-only).

    python scripts/p4_subset_score.py <checkpoint.pkl> <summary_latest.json> K <out.json>

Why: an n=1,000 run and an n=300 run on --select head share their first 300
images, seed and per-image computation (nothing per image depends on --n), so
the first 300 images of the 1,000-image checkpoint are the paired baseline for
a 300-image ablation without re-running the attack. The checkpoint is only
read; the COCO scoring is phase4/p4_core.score_detection, the same call the
run itself made, so on K = all images this reproduces the run's own numbers
(checked by --self-check). The output file name must start with "summary"
(results/ .gitignore policy).
"""
import json
import os
import pickle
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "phase4"))
sys.path.insert(0, REPO)

import p4_core as core                     # noqa: E402
import p4_yolo as ya                       # noqa: E402
from p4_core import rt                     # noqa: E402


def main():
    ckpt_path, summary_path, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    import numpy, torch, PIL.Image, phase3_common            # noqa: E401
    rt.np, rt.torch, rt.Image, rt.pc = numpy, torch, PIL.Image, phase3_common
    from pycocotools.coco import COCO

    run = json.load(open(summary_path))
    with open(ckpt_path, "rb") as f:
        saved = pickle.load(f)
    buckets = saved["buckets"]
    ids = run["image_ids"][:k]                 # --select head order
    missing = [i for i in ids if i not in buckets]
    assert not missing, f"{len(missing)} of the first {k} images are not in the checkpoint"
    sub = SimpleNamespace(buckets={i: buckets[i] for i in ids})

    target = run["target"]
    ensembles, branches, paper = ya.resolve_target(target)
    args = SimpleNamespace(nms_iou=run["defense"]["nms_iou"], output_dir=os.path.dirname(out),
                           keep_eval_dumps=False, verbose_eval=False)
    coco_gt = COCO(os.path.join(REPO, "annotations", "instances_val2017.json"))
    OBL, ADP = ya.OBLIVIOUS_TAG, ya.ADAPTIVE_TAG
    _, cond = core.score_detection(args, sub, branches, ensembles, coco_gt,
                                   attack_tags=[OBL, ADP])
    head = {
        "clean_mAP": cond["clean"]["mAP"],
        "clean_defended_mAP": cond[f"clean+{target}"]["mAP"],
        "oblivious_attacked_undefended_mAP": cond[OBL]["mAP"],
        "oblivious_attacked_defended_mAP": cond[f"{OBL}+{target}"]["mAP"],
        "adaptive_attacked_undefended_mAP": cond[ADP]["mAP"],
        "adaptive_attacked_defended_mAP": cond[f"{ADP}+{target}"]["mAP"],
    }
    payload = {"source_checkpoint": os.path.relpath(ckpt_path, REPO),
               "source_summary": os.path.relpath(summary_path, REPO),
               "first_k_images": k, "image_ids_sha1_16": core.ids_digest(ids),
               "target": target, "attack_flags": run.get("attack_flags"),
               "attack": run.get("attack"), "headline": head, "conditions": cond}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"k": k, "ids": payload["image_ids_sha1_16"], **{a: round(b, 4) for a, b in head.items()}}, indent=1))
    if k == len(run["image_ids"]):
        for key, v in head.items():
            assert abs(v - run["headline"][key]) < 1e-9, f"self-check failed on {key}: {v} vs {run['headline'][key]}"
        print("[self-check] K = all images reproduces the run's own headline exactly")


if __name__ == "__main__":
    main()
