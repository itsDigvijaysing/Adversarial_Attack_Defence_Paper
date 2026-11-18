#!/usr/bin/env python3
"""Paired image-level bootstrap for a finished oblivious_survey run (read-only).

    python scripts/p4_boot_survey.py <checkpoint.pkl> <summary_latest.json> <out.json>
        [--B 1000] [--seed 42] [--workers 8]

For every ensemble: (ensemble - its best member) under the oblivious attack,
(ensemble - Median) under the attack, and the ensemble's clean cost, each with
a 95% percentile interval over images. Same method as phase4/p4_rescore.py
(whose COCOeval-compaction and vectorised accumulate() it imports): one
(B x N) matrix of image positions shared by every condition, so differences
are paired; per condition the fast engine must reproduce COCOeval's mAP on
the full set (asserted). The checkpoint is only read; the COCO detections are
assembled with phase3_common.assemble_results, the call the run itself made,
so the point estimates equal the run's summary (asserted). The output file
name must start with "summary" (results/ .gitignore policy).
"""
import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "phase4"))
sys.path.insert(0, REPO)

import numpy as np                                        # noqa: E402
import p4_rescore as rs                                   # noqa: E402

OBL = "pgd_oblivious"
_G = {}


def _init(dets_by_tag, ann, ids, idx):
    from pycocotools.coco import COCO
    with rs.quiet():
        _G["gt"] = COCO(ann)
    _G.update(dets=dets_by_tag, ids=ids, idx=idx)


def _boot(tag):
    from pycocotools.cocoeval import COCOeval
    t0 = time.time()
    dets, ids, idx = _G["dets"][tag], _G["ids"], _G["idx"]
    with rs.quiet():
        dt = _G["gt"].loadRes(dets)
        ev = COCOeval(_G["gt"], dt, "bbox")
        ev.params.imgIds = list(ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    compact = rs.compact_from_eval(ev)
    m, _ = rs.fast_resample(compact, np.arange(len(ids)))
    assert abs(m - ev.stats[0]) < 1e-9, (tag, m, ev.stats[0])
    boot = np.array([rs.fast_resample(compact, idx[b])[0]
                     for b in range(idx.shape[0])])
    return tag, float(ev.stats[0]), boot, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"), ap.add_argument("summary"), ap.add_argument("out")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    assert os.path.basename(a.out).startswith("summary"), "results/ naming"
    import phase3_common as pc

    run = json.load(open(a.summary))
    with open(a.ckpt, "rb") as f:
        buckets = pickle.load(f)["buckets"]
    ids = list(run["image_ids"])
    assert sorted(buckets) == sorted(ids), "checkpoint and summary disagree"
    names, ensembles = run["branches"], run["ensembles"]
    allres = pc.assemble_results(buckets, defense_names=names,
                                 attack_tags=[OBL], ensembles=ensembles,
                                 nms_iou=run["config"]["nms_iou"])
    tags = ["clean", OBL]
    for e, members in ensembles.items():
        tags += [f"clean+{e}", f"{OBL}+{e}"] + [f"{OBL}+{m}" for m in members]
    tags += [f"{OBL}+median"]
    tags = list(dict.fromkeys(tags))
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, len(ids), size=(a.B, len(ids)))
    ann = os.path.join(REPO, "annotations", "instances_val2017.json")
    print(f"[boot] {len(tags)} conditions, B={a.B}, N={len(ids)}, "
          f"{a.workers} workers")
    t0 = time.time()
    with Pool(a.workers, initializer=_init,
              initargs=({t: allres[t] for t in tags}, ann, ids, idx)) as pool:
        res = {}
        for tag, point, boot, sec in pool.imap_unordered(_boot, tags):
            want = run["conditions"][tag]["mAP"]
            assert abs(point - want) < 1e-9, f"{tag}: {point} vs summary {want}"
            res[tag] = (point, boot)
            print(f"[boot] {tag:<40} mAP {point:.4f}  ({sec:.0f}s)", flush=True)
    out = {"source_checkpoint": os.path.relpath(a.ckpt, REPO),
           "source_summary": os.path.relpath(a.summary, REPO),
           "B": a.B, "seed": a.seed, "ci": 95.0, "n_images": len(ids),
           "method": "paired image-level bootstrap, percentile CI; fast "
                     "accumulate() replica of phase4/p4_rescore.py",
           "ensembles": {}}
    for e, members in ensembles.items():
        best = max(members, key=lambda m: res[f"{OBL}+{m}"][0])
        d = lambda x, y: rs.paired_delta(res[x][1], res[y][1],    # noqa: E731
                                         res[x][0], res[y][0], 95.0)
        out["ensembles"][e] = {
            "best_member_under_attack": best,
            "ensemble_minus_best_member_attacked": d(f"{OBL}+{e}", f"{OBL}+{best}"),
            "ensemble_minus_median_attacked": d(f"{OBL}+{e}", f"{OBL}+median"),
            "clean_cost": d(f"clean+{e}", "clean"),
        }
        r = out["ensembles"][e]
        print(f"  {e:<26} vs {best:<9} "
              f"{r['ensemble_minus_best_member_attacked']['point']:+.4f} "
              f"[{r['ensemble_minus_best_member_attacked']['lo']:+.4f}, "
              f"{r['ensemble_minus_best_member_attacked']['hi']:+.4f}]  "
              f"clean cost {r['clean_cost']['point']:+.4f} "
              f"[{r['clean_cost']['lo']:+.4f}, {r['clean_cost']['hi']:+.4f}]")
    out["seconds"] = time.time() - t0
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
