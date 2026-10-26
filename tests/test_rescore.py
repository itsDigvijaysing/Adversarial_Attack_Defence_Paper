"""Offline checks for phase4/p4_rescore.py. No GPU, no torch, no COCO data.

    python3 tests/test_rescore.py                                   # numpy only
    /home/king/miniconda3/envs/vlm_ftune/bin/python tests/test_rescore.py  # + pycocotools

Covers: the CLI and output names, the score-free greedy matcher on hand-made
boxes, the vectorised accumulate() replica against a line-by-line port of
pycocotools' loop, and (when pycocotools is importable) the duplicated-id
bootstrap against COCOeval on a dataset whose duplicated images are real
copies, plus one end-to-end run of main() on a synthetic run directory.
Inputs are SYNTHETIC: no number printed here is a measurement, and nothing is
written into results/ (temp dirs only). pycocotools parts are SKIPPED, not
failed, when the module is missing.
"""
import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE4 = os.path.join(REPO, "phase4")
sys.path.insert(0, PHASE4)

import p4_logging as rl  # noqa: E402
import p4_rescore as rs  # noqa: E402

try:
    import pycocotools  # noqa: F401
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAVE_PCT = True
except ImportError:
    HAVE_PCT = False

PASS, FAIL, SKIP = [], [], []
EM_DASH = chr(0x2014)   # spelled as a code point so this file has none


def check(name, fn, needs_pct=False):
    if needs_pct and not HAVE_PCT:
        SKIP.append(name)
        print(f"  SKIP  {name} (pycocotools not importable)")
        return
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception:  # noqa: BLE001
        FAIL.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()


# ----------------------------------------------------------------------
# CLI and output names
# ----------------------------------------------------------------------
def t_cli_defaults():
    a = rs.parse_args([])
    assert a.B == 1000 and a.seed == 42 and a.ci == 95.0 and a.n == 1000
    assert a.models == list(rs.MODELS) and a.attacks == list(rs.ATTACKS)
    assert a.partial is False
    assert a.output_dir == rs.DEFAULT_OUTPUT_DIR
    assert a.output_dir.endswith(os.path.join("results", "phase4",
                                              "rescore_phase3_v2"))
    assert a.engine == "fast" and a.workers >= 1 and a.tolerance == 1e-4


def t_cli_subset_is_partial():
    a = rs.parse_args(["--models", "yolo", "--attacks", "pgd", "--B", "50",
                       "--only", "clean", "clean+median"])
    assert a.B == 50 and a.only == ["clean", "clean+median"]
    assert a.partial is True
    assert a.output_dir == os.path.join(rs.DEFAULT_OUTPUT_DIR, "smoke_test")
    b = rs.parse_args(["--attacks", "fgsm", "--output-dir", "/x/y"])
    assert b.partial is True and b.output_dir == "/x/y"
    c = rs.parse_args(["--workers", "0"])
    assert c.workers == 1


def t_cli_rejects_bad_values():
    for bad in (["--B", "0"], ["--ci", "40"], ["--ci", "100"], ["--n", "0"],
                ["--models", "resnet"], ["--diag-resamples", "-1"],
                ["--task-timeout", "0"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                rs.parse_args(bad)
            except SystemExit:
                continue
        raise AssertionError(f"accepted {bad}")


def t_output_names_start_with_summary():
    paths = rs.output_paths("/tmp/out", "20261024-120000")
    assert set(paths) == {"json", "json_latest", "md", "md_latest", "log"}
    for key, p in paths.items():
        base = os.path.basename(p)
        assert os.path.dirname(p) == "/tmp/out"
        if key == "log":
            assert base == "run.log"
        else:
            assert base.startswith("summary"), base
            assert base.startswith("summary_rescore_phase3_v2_"), base
        assert rl.is_committed(base), base
    assert paths["json"].endswith("_20261024-120000.json")
    assert paths["json_latest"].endswith("_latest.json")


def t_tag_to_file():
    assert rs.tag_to_file("clean") == "clean.json"
    assert rs.tag_to_file("clean+median") == "clean_median.json"
    assert (rs.tag_to_file("pgd_eps0.03+ens_jpeg_median_gaussian")
            == "pgd_eps0.03_ens_jpeg_median_gaussian.json")


# ----------------------------------------------------------------------
# Score-free matching
# ----------------------------------------------------------------------
def t_iou_values():
    a = [[0, 0, 10, 10]]
    assert np.isclose(rs.box_iou_xywh(a, a)[0, 0], 1.0)
    # half overlap: inter 50, union 150
    assert np.isclose(rs.box_iou_xywh(a, [[5, 0, 10, 10]])[0, 0], 1 / 3)
    assert rs.box_iou_xywh(a, [[20, 20, 5, 5]])[0, 0] == 0.0
    assert rs.box_iou_xywh(np.zeros((0, 4)), a).shape == (0, 1)
    # crowd rule: intersection over the DETECTION area
    assert np.isclose(rs.box_iof_xywh([[0, 0, 4, 4]], [[0, 0, 100, 100]])[0, 0],
                      1.0)


def t_match_is_iou_greedy_not_score_greedy():
    g = [[0, 0, 10, 10], [5, 0, 10, 10]]
    # det0 (high score) overlaps g0 at 0.667 and g1 at 0.538; det1 (low
    # score) sits exactly on g0 and overlaps g1 at 0.333. Score-first greedy
    # would give det0 -> g0 and leave det1 unmatched (1 TP). IoU-first gives
    # det1 -> g0, then det0 -> g1 (2 TP).
    d = [[2, 0, 10, 10], [0, 0, 10, 10]]
    iou = rs.box_iou_xywh(d, g)
    assert np.allclose(iou, [[80 / 120, 70 / 130], [1.0, 50 / 150]])
    pairs, ign = rs.match_greedy(d, [0.9, 0.1], g)
    assert pairs == [(1, 0), (0, 1)], pairs
    assert not ign.any()
    assert rs.match_counts(d, [0.9, 0.1], g) == (2, 0, 0, 0)


def t_match_one_to_one_and_threshold():
    g = [[0, 0, 10, 10]]
    tp, fp, fn, ign = rs.match_counts([[0, 0, 10, 10], [1, 0, 10, 10]],
                                      [0.2, 0.9], g)
    assert (tp, fp, fn, ign) == (1, 1, 0, 0)
    # IoU exactly 0.5 counts (>=): [0,0,10,10] vs [0,0,10,20] -> 100 / 200
    assert rs.match_counts([[0, 0, 10, 20]], [0.5], g) == (1, 0, 0, 0)
    assert rs.match_counts([[0, 0, 10, 21]], [0.5], g) == (0, 1, 1, 0)
    # empty sides
    assert rs.match_counts(np.zeros((0, 4)), [], g) == (0, 0, 1, 0)
    assert rs.match_counts([[0, 0, 5, 5]], [0.3], np.zeros((0, 4))) == \
        (0, 1, 0, 0)


def t_match_score_breaks_exact_ties():
    g = [[0, 0, 10, 10]]
    d = [[0, 0, 10, 12], [0, 0, 10, 12]]          # identical IoU
    pairs, _ = rs.match_greedy(d, [0.3, 0.8], g)
    assert pairs == [(1, 0)], pairs                # higher score wins the tie
    pairs, _ = rs.match_greedy(d, [0.8, 0.3], g)
    assert pairs == [(0, 0)], pairs


def t_match_crowd_ignored():
    g = [[0, 0, 10, 10]]
    crowd = [[50, 50, 40, 40]]
    d = [[0, 0, 10, 10], [55, 55, 10, 10], [200, 200, 5, 5]]
    pairs, ign = rs.match_greedy(d, [0.9, 0.8, 0.7], g, crowd)
    assert pairs == [(0, 0)] and ign.tolist() == [False, True, False]
    assert rs.match_counts(d, [0.9, 0.8, 0.7], g, crowd) == (1, 1, 0, 1)


def t_score_free_counts_per_class_and_threshold():
    ids = [1, 2]
    gt_sf = {(1, 7): (np.array([[0, 0, 10, 10]], float), np.zeros((0, 4))),
             (2, 7): (np.array([[0, 0, 10, 10]], float), np.zeros((0, 4)))}
    dets = [
        {"image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10], "score": .9},
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": .8},
        {"image_id": 2, "category_id": 7, "bbox": [0, 0, 10, 10], "score": .1},
        {"image_id": 99, "category_id": 7, "bbox": [0, 0, 1, 1], "score": .9},
    ]
    c = rs.score_free_counts(dets, gt_sf, ids, 0.5)
    # image 1: class 7 TP, class 3 FP (wrong class never matches); image 2 TP;
    # image 99 is outside the scored ids and is dropped.
    assert c["tp"].tolist() == [1, 1] and c["fp"].tolist() == [1, 0]
    assert c["fn"].tolist() == [0, 0]
    c = rs.score_free_counts(dets, gt_sf, ids, 0.5, score_thr=0.25)
    assert c["tp"].tolist() == [1, 0] and c["fn"].tolist() == [0, 1]
    p, r, f = rs.prf(2, 1, 0)
    assert np.isclose(p, 2 / 3) and np.isclose(r, 1) and np.isclose(f, 0.8)
    p, r, f = rs.prf(0, 0, 0)                     # no division by zero
    assert p == 0 and r == 0 and f == 0


def t_score_free_bootstrap_weights():
    counts = {"tp": np.array([1, 2, 0]), "fp": np.array([0, 1, 3]),
              "fn": np.array([1, 0, 1])}
    idx = np.array([[0, 1, 2], [0, 0, 2], [1, 1, 1]])
    W = rs.multiplicity_matrix(idx, 3)
    assert W.tolist() == [[1, 1, 1], [2, 0, 1], [0, 3, 0]]
    b = rs.sf_boot(counts, W)
    f_full = rs.prf(3, 4, 2)[2]
    assert np.isclose(b["f1"][0], f_full)          # identity resample
    assert np.isclose(b["precision"][2], 6 / 9)    # image 1 three times


# ----------------------------------------------------------------------
# Bootstrap statistics
# ----------------------------------------------------------------------
def t_paired_delta_and_bc():
    rng = np.random.default_rng(0)
    b = rng.normal(0.3, 0.01, 2000)
    d = rs.paired_delta(b + 0.05, b, 0.35, 0.30, 95)
    assert np.isclose(d["point"], 0.05)
    assert np.isclose(d["lo"], 0.05) and np.isclose(d["hi"], 0.05)
    assert d["excludes_0"] and d["share_le0"] == 0.0
    # symmetric and centred on the point: BC == plain percentile (approx)
    x = rng.normal(0.0, 1.0, 20000)
    lo, hi = rs.percentile_ci(x, 95)
    blo, bhi = rs.bc_percentile_ci(x, float(np.median(x)), 95)
    assert abs(lo - blo) < 0.05 and abs(hi - bhi) < 0.05
    # an upward-biased bootstrap pulls the BC interval down
    blo, bhi = rs.bc_percentile_ci(x + 0.5, 0.0, 95)
    lo2, hi2 = rs.percentile_ci(x + 0.5, 95)
    assert blo < lo2 and bhi < hi2
    s = rs.boot_summary(0.0, x + 0.5, 95)
    assert abs(s["bias"] - 0.5) < 0.05 and s["bias_over_se"] > 0.4


# ----------------------------------------------------------------------
# Fast accumulate replica vs a literal port of pycocotools' loop
# ----------------------------------------------------------------------
REC_THRS = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1,
                       endpoint=True)
IOU_THRS = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1,
                       endpoint=True)


def ref_precision(eval_imgs, K, A, I, a0, pos, maxDet, envelope=True):
    """Line-by-line port of COCOeval.accumulate() for one area / maxDet.

    envelope=False drops the running-maximum loop (the diagnostic variant).
    """
    T, R = len(IOU_THRS), len(REC_THRS)
    precision = -np.ones((T, R, K))
    for k in range(K):
        E = [eval_imgs[k * A * I + a0 * I + i] for i in pos]
        E = [e for e in E if e is not None]
        if len(E) == 0:
            continue
        dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])
        inds = np.argsort(-dtScores, kind="mergesort")
        dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E],
                             axis=1)[:, inds]
        dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E],
                              axis=1)[:, inds]
        gtIg = np.concatenate([e["gtIgnore"] for e in E])
        npig = np.count_nonzero(gtIg == 0)
        if npig == 0:
            continue
        tps = np.logical_and(dtm, np.logical_not(dtIg))
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
        tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
        fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
        for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            nd = len(tp)
            rc = tp / npig
            pr = (tp / (fp + tp + np.spacing(1))).tolist()
            q = np.zeros((R,)).tolist()
            for i in range(nd - 1, 0, -1) if envelope else ():
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]
            ii = np.searchsorted(rc, REC_THRS, side="left")
            try:
                for ri, pi in enumerate(ii):
                    q[ri] = pr[pi]
            except IndexError:
                pass
            precision[t, :, k] = np.array(q)
    return precision


def synthetic_eval_imgs(rng, K=4, A=2, I=9, maxDet=5):
    """Random evalImgs-shaped entries (some None, some GT-only/det-only)."""
    T = len(IOU_THRS)
    out = []
    gid = 1
    for _k in range(K):
        for _a in range(A):
            for _i in range(I):
                if rng.random() < 0.2:
                    out.append(None)
                    continue
                D = int(rng.integers(0, 8))
                G = int(rng.integers(0, 4))
                if D == 0 and G == 0:
                    out.append(None)
                    continue
                scores = np.sort(rng.random(D))[::-1][:maxDet + 2]
                # matches get weaker as the IoU threshold rises
                base = rng.random(D) < 0.6
                dtm = np.zeros((T, D))
                for t in range(T):
                    hit = base & (rng.random(D) < 1 - t / (T + 2))
                    dtm[t, hit] = np.arange(gid, gid + hit.sum())
                gid += D
                dtig = np.zeros((T, D), bool)
                dtig[:, rng.random(D) < 0.1] = True
                gtig = (rng.random(G) < 0.2).astype(int)
                out.append({"dtScores": list(scores), "dtMatches": dtm,
                            "dtIgnore": dtig, "gtIgnore": gtig})
    return out, K, A, I


def t_fast_matches_reference_loop():
    rng = np.random.default_rng(7)
    maxDet = 5
    for trial in range(4):
        ev, K, A, I = synthetic_eval_imgs(rng, maxDet=maxDet)
        for a0 in range(A):
            compact = rs.compact_from_evalimgs(ev, K, A, I, a0, len(IOU_THRS),
                                               maxDet, REC_THRS)
            pos_list = [np.arange(I), rng.permutation(I),
                        rng.integers(0, I, I), np.array([0, 0, 0, 3]),
                        rng.integers(0, I, 2 * I)]
            for pos in pos_list:
                want = ref_precision(ev, K, A, I, a0, pos, maxDet)
                got = rs.fast_precision(compact, pos)
                assert np.array_equal(want, got), (trial, a0, pos.tolist(),
                                                   np.abs(want - got).max())
                assert rs.summarize_precision(got) == \
                    rs.summarize_precision(want)
                want0 = ref_precision(ev, K, A, I, a0, pos, maxDet,
                                      envelope=False)
                got0 = rs.fast_precision(compact, pos, envelope=False)
                assert np.array_equal(want0, got0)
                assert (got0 <= got).all()        # the envelope only raises


# ----------------------------------------------------------------------
# pycocotools: duplicated-id resample == COCOeval on real copies
# ----------------------------------------------------------------------
CATS = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"},
        {"id": 3, "name": "c"}]


def make_gt(rng, n_img=6):
    images = [{"id": i, "width": 200, "height": 200, "file_name": f"{i}.jpg"}
              for i in range(1, n_img + 1)]
    anns, aid = [], 1
    for im in images:
        for _ in range(int(rng.integers(1, 6))):
            x, y = rng.uniform(0, 140, 2)
            w, h = rng.uniform(8, 60, 2)
            anns.append({"id": aid, "image_id": im["id"],
                         "category_id": int(rng.integers(1, 4)),
                         "bbox": [float(x), float(y), float(w), float(h)],
                         "area": float(w * h),
                         "iscrowd": int(rng.random() < 0.1)})
            aid += 1
    return {"images": images, "annotations": anns, "categories": CATS}


def make_dets(rng, gt, jitter=4.0, p_keep=0.8, n_fp=3):
    dets = []
    for a in gt["annotations"]:
        if a["iscrowd"] or rng.random() > p_keep:
            continue
        x, y, w, h = a["bbox"]
        dx, dy, dw, dh = rng.normal(0, jitter, 4)
        cat = a["category_id"] if rng.random() > 0.1 else int(
            rng.integers(1, 4))
        dets.append({"image_id": a["image_id"], "category_id": cat,
                     "bbox": [x + dx, y + dy, max(2.0, w + dw),
                              max(2.0, h + dh)]})
    for im in gt["images"]:
        for _ in range(int(rng.integers(0, n_fp + 1))):
            x, y = rng.uniform(0, 150, 2)
            w, h = rng.uniform(5, 50, 2)
            dets.append({"image_id": im["id"],
                         "category_id": int(rng.integers(1, 4)),
                         "bbox": [float(x), float(y), float(w), float(h)]})
    # distinct scores, so the only ties a resample creates are copies of the
    # same detection (whose matches are identical, so their order is moot)
    scores = rng.permutation(len(dets)) / len(dets) * 0.9 + 0.05
    for d, s in zip(dets, scores):
        d["score"] = float(s)
    return dets


def coco_from(ds):
    gt = COCO()
    gt.dataset = copy.deepcopy(ds)
    with rs.quiet():
        gt.createIndex()
    return gt


def coco_stats(ds, dets, img_ids):
    gt = coco_from(ds)
    with rs.quiet():
        dt = gt.loadRes(copy.deepcopy(dets))
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(img_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def copies_dataset(ds, dets, ids, pos):
    """Position j of the resample becomes a real image with id j + 1."""
    by_img = {}
    for a in ds["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    dby = {}
    for d in dets:
        dby.setdefault(d["image_id"], []).append(d)
    images, anns, new_dets, aid = [], [], [], 1
    for j, p in enumerate(pos):
        old, new = ids[int(p)], j + 1
        images.append({"id": new, "width": 200, "height": 200,
                       "file_name": f"{new}.jpg"})
        for a in by_img.get(old, []):
            b = dict(a, id=aid, image_id=new)
            aid += 1
            anns.append(b)
        for d in dby.get(old, []):
            new_dets.append(dict(d, image_id=new))
    return ({"images": images, "annotations": anns, "categories": CATS},
            new_dets, [im["id"] for im in images])


def t_bootstrap_duplicates_equal_real_copies():
    rng = np.random.default_rng(3)
    ds = make_gt(rng, n_img=7)
    dets = make_dets(rng, ds)
    ids = sorted(im["id"] for im in ds["images"])
    ev = coco_stats(ds, dets, ids)
    compact = rs.compact_from_eval(ev)
    N = len(ids)
    # (1) every id once, in order and shuffled: equals the point estimate
    for pos in (np.arange(N), rng.permutation(N)):
        fm, f50 = rs.fast_resample(compact, pos)
        lm, l50 = rs.literal_resample(ev, pos)
        assert abs(fm - ev.stats[0]) < 1e-12 and abs(f50 - ev.stats[1]) < 1e-12
        assert abs(lm - ev.stats[0]) < 1e-12 and abs(l50 - ev.stats[1]) < 1e-12
    # (2) resamples with duplicates and omissions: equals COCOeval on a
    #     dataset in which the duplicated images are real copies
    for pos in (np.array([0, 0, 2, 3, 3, 3, 6]), rng.integers(0, N, N),
                rng.integers(0, N, N), np.array([5, 5, 5, 5, 5, 5, 5])):
        cds, cdets, cids = copies_dataset(ds, dets, ids, pos)
        want = coco_stats(cds, cdets, cids).stats[:2]
        fm, f50 = rs.fast_resample(compact, pos)
        lm, l50 = rs.literal_resample(ev, pos)
        assert abs(fm - want[0]) < 1e-12 and abs(f50 - want[1]) < 1e-12, \
            (pos.tolist(), fm, want[0])
        assert abs(lm - want[0]) < 1e-12 and abs(l50 - want[1]) < 1e-12


def t_evaluate_condition_worker():
    rng = np.random.default_rng(11)
    ds = make_gt(rng, n_img=8)
    dets = make_dets(rng, ds)
    ids = sorted(im["id"] for im in ds["images"])
    tmp = tempfile.mkdtemp(prefix="p4_rescore_test_")
    try:
        path = os.path.join(tmp, "clean.json")
        with open(path, "w") as f:
            json.dump(dets, f)
        B = 25
        idx = np.random.default_rng(0).integers(0, len(ids), (B, len(ids)))
        gt = coco_from(ds)
        ctx = {"gt": gt, "ids": ids, "idset": set(ids), "idx": idx,
               "gt_sf": rs.gt_by_img_cat(gt.dataset["annotations"]),
               "sf_iou": 0.5, "yolo_sf_conf": 0.25, "engine": "fast",
               "verify_resamples": 5}
        res = rs.evaluate_condition({"key": "yolo/x", "model": "yolo",
                                     "path": path}, ctx)
        ev = coco_stats(ds, dets, ids)
        assert abs(res["point"]["mAP"] - ev.stats[0]) < 1e-12
        assert res["boot"]["mAP"].shape == (B,)
        assert res["engine_point_diff"] == 0.0
        assert len(res["engine_resample_diffs"]) == 5
        assert max(res["engine_resample_diffs"]) < 1e-12
        assert set(res["score_free_counts"]) == {"all", "conf0.25"}
        ctx["engine"] = "pycocotools"
        res2 = rs.evaluate_condition({"key": "yolo/x", "model": "yolo",
                                      "path": path}, ctx)
        assert np.allclose(res2["boot"]["mAP"], res["boot"]["mAP"],
                           atol=1e-12, rtol=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_main_end_to_end():
    """main() on a synthetic results_phase3_yolo_pgd_v2/ run directory."""
    rng = np.random.default_rng(5)
    ds = make_gt(rng, n_img=10)
    ids = sorted(im["id"] for im in ds["images"])
    tmp = tempfile.mkdtemp(prefix="p4_rescore_e2e_")
    try:
        ann = os.path.join(tmp, "instances_val2017.json")
        with open(ann, "w") as f:
            json.dump(ds, f)
        run_dir = os.path.join(tmp, "in", "results_phase3_yolo_pgd_v2")
        os.makedirs(run_dir)
        defenses = {"median": "solo", "jpeg": "solo",
                    "ens_jpeg_median_gaussian": "ENSEMBLE"}
        tags = (["clean"] + [f"clean+{d}" for d in defenses] + ["pgd_eps0.03"]
                + [f"pgd_eps0.03+{d}" for d in defenses])
        stats = {}
        for t in tags:
            dets = make_dets(rng, ds, jitter=6.0 if "pgd" in t else 3.0,
                             p_keep=0.5 if t == "pgd_eps0.03" else 0.8)
            with open(os.path.join(run_dir, rs.tag_to_file(t)), "w") as f:
                json.dump(dets, f)
            stats[t] = coco_stats(ds, dets, ids).stats
        att = float(stats["pgd_eps0.03"][0])
        summary = {
            "model": "synthetic", "attack": "pgd_eps0.03", "epsilon": 0.03,
            "num_images": len(ids), "clean_mAP": float(stats["clean"][0]),
            "clean_AP50": float(stats["clean"][1]),
            "attacked_mAP": att,
            "attack_damage": float(stats["clean"][0]) - att,
            "defenses": {d: {"mAP": float(stats[f"pgd_eps0.03+{d}"][0]),
                             "AP50": float(stats[f"pgd_eps0.03+{d}"][1]),
                             "recovery": float(
                                 stats[f"pgd_eps0.03+{d}"][0]) - att,
                             "kind": k} for d, k in defenses.items()}}
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f)
        out = os.path.join(tmp, "out")
        argv = ["--models", "yolo", "--attacks", "pgd", "--B", "30",
                "--n", str(len(ids)), "--input-root", os.path.join(tmp, "in"),
                "--ann-file", ann, "--output-dir", out,
                "--task-timeout", "120",        # a stall fails, never hangs
                "--workers", "2"]
        with contextlib.redirect_stdout(io.StringIO()):
            code = rs.main(argv)
        assert code == 0, code
        files = sorted(os.listdir(out))
        assert "run.log" in files and len(files) == 5, files
        assert all(f.startswith("summary") for f in files if f != "run.log")
        jl = [f for f in files if f.endswith("_latest.json")][0]
        with open(os.path.join(out, jl)) as f:
            P = json.load(f)
        assert P["partial"] is True
        assert P["agreement"]["n_mismatch"] == 0
        # 4 top-level (clean mAP/AP50, attacked, damage) + 3 per defense
        assert P["agreement"]["n_checked"] == 4 + 3 * len(defenses)
        assert P["image_id_check"]["yolo"]["subset_of_scored_ids"]
        # the JSON keeps 7 significant digits, hence 1e-6 below
        cc = {r["defense"]: r for r in P["clean_cost"]["yolo"]}
        assert abs(cc["median"]["delta"]["mAP"]["point"]
                   - (stats["clean+median"][0] - stats["clean"][0])) < 1e-6
        bvm = P["best_ensemble_vs_median"][0]
        assert bvm["best_ensemble"] == "ens_jpeg_median_gaussian"
        assert abs(bvm["delta"]["point"]
                   - (stats["pgd_eps0.03+ens_jpeg_median_gaussian"][0]
                      - stats["pgd_eps0.03+median"][0])) < 1e-6
        hl = {h["defense"]: h for h in P["headline_check"]}
        assert set(hl) == set(defenses)
        for dname, h in hl.items():
            rec = stats[f"pgd_eps0.03+{dname}"][0] - stats["pgd_eps0.03"][0]
            assert h["settings"] == 1 and h["point_pos"] == int(rec > 0)
        bd = P["bootstrap_sanity"]["decomposition"]
        assert np.isclose(bd["median_total"],
                          np.median([c["bias_decomposition"]["total"]
                                     for c in P["conditions"]]), atol=1e-6)
        for c in P["conditions"]:
            d = c["bias_decomposition"]
            assert abs(d["total"] - d["composition"] - d["within_class"]) \
                < 1e-6
            assert abs(c["bootstrap"]["mAP"]["bias"] - d["total"]) < 1e-6
        md = open(os.path.join(out, jl.replace(".json", ".md"))).read()
        assert "PARTIAL RUN" in md and "## How to read" in md
        assert "## Headline check" in md
        assert EM_DASH not in md                  # no em dash in outputs
        # a wrong committed value is reported and turns the exit code to 2
        summary["clean_mAP"] += 0.01
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f)
        with contextlib.redirect_stdout(io.StringIO()):
            code = rs.main(argv[:-1] + ["1"])
        assert code == 2, code
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_class_ap_and_bias_decomposition():
    rng = np.random.default_rng(21)
    ev, K, A, I = synthetic_eval_imgs(rng, K=6, I=12, maxDet=5)
    compact = rs.compact_from_evalimgs(ev, K, A, I, 0, len(IOU_THRS), 5,
                                       REC_THRS)
    full = rs.fast_precision(compact, np.arange(I))
    ap = rs.class_ap(full)
    point = rs.summarize_precision(full)[0]
    assert np.isclose(np.nanmean(ap), point)       # mean of classes == mAP
    # identity resamples: no class is absent, nothing to decompose
    idx = np.tile(np.arange(I), (4, 1))
    boot = np.array([rs.summarize_precision(rs.fast_precision(compact, p))[0]
                     for p in idx])
    filled = np.array([rs.filled_map(rs.class_ap(rs.fast_precision(
        compact, p)), ap) for p in idx])
    d = rs.bias_decomposition(compact, idx, point, boot, filled, ap, 4)
    assert abs(d["total"]) < 1e-12 and abs(d["composition"]) < 1e-12
    assert abs(d["envelope_share_diag"]) < 1e-12 and d["n_diag"] == 4
    # random resamples: total == composition + within_class, exactly
    idx = rng.integers(0, I, (30, I))
    idx[0] = 0                                     # one image only: classes drop
    boot, filled = [], []
    for p in idx:
        pr = rs.fast_precision(compact, p)
        boot.append(rs.summarize_precision(pr)[0])
        filled.append(rs.filled_map(rs.class_ap(pr), ap))
    boot, filled = np.array(boot), np.array(filled)
    d = rs.bias_decomposition(compact, idx, point, boot, filled, ap, 10)
    assert np.isclose(d["total"], d["composition"] + d["within_class"])
    assert d["n_diag"] == 10 and "within_class_no_envelope_diag" in d
    assert rs.bias_decomposition(compact, idx, point, boot, filled, ap,
                                 0)["n_diag"] == 0
    # a class absent from the resample is put back at its full-sample AP
    ap_b = ap.copy()
    k = int(np.flatnonzero(~np.isnan(ap))[0])
    ap_b[k] = np.nan
    assert np.isclose(rs.filled_map(ap_b, ap), np.nanmean(ap))


def t_round_floats():
    x = {"a": 0.36053955430077717, "b": [1e-9 + 1e-17, 3, True, None],
         "c": {"d": float("nan")}, "e": 0.0}
    y = rs.round_floats(x)
    assert y["a"] == 0.3605396 and y["b"][1:] == [3, True, None]
    assert y["b"][0] == 1e-9 and y["e"] == 0.0 and np.isnan(y["c"]["d"])


def t_source_has_no_em_dash():
    for p in (os.path.join(PHASE4, "p4_rescore.py"), os.path.abspath(__file__)):
        with open(p, encoding="utf-8") as f:
            assert EM_DASH not in f.read(), p


if __name__ == "__main__":
    print(f"p4_rescore offline checks (pycocotools "
          f"{'available' if HAVE_PCT else 'MISSING: those checks skip'})")
    check("cli defaults", t_cli_defaults)
    check("cli subset is partial -> smoke_test/", t_cli_subset_is_partial)
    check("cli rejects bad values", t_cli_rejects_bad_values)
    check("output names start with 'summary'",
          t_output_names_start_with_summary)
    check("tag -> dump file name", t_tag_to_file)
    check("iou / iof values", t_iou_values)
    check("matching: IoU-greedy, not score-greedy",
          t_match_is_iou_greedy_not_score_greedy)
    check("matching: one-to-one, IoU >= thr", t_match_one_to_one_and_threshold)
    check("matching: score only breaks exact ties",
          t_match_score_breaks_exact_ties)
    check("matching: crowd regions ignored", t_match_crowd_ignored)
    check("score-free counts: per class, threshold, outside ids",
          t_score_free_counts_per_class_and_threshold)
    check("score-free bootstrap: multiplicity weights",
          t_score_free_bootstrap_weights)
    check("paired delta + bias-corrected CI", t_paired_delta_and_bc)
    check("fast accumulate == port of pycocotools loop (dup ids)",
          t_fast_matches_reference_loop)
    check("class AP + bias decomposition (composition / within-class)",
          t_class_ap_and_bias_decomposition)
    check("json float rounding", t_round_floats)
    check("pycocotools: dup-id resample == COCOeval on real copies",
          t_bootstrap_duplicates_equal_real_copies, needs_pct=True)
    check("pycocotools: evaluate_condition worker, both engines",
          t_evaluate_condition_worker, needs_pct=True)
    check("pycocotools: main() end to end + mismatch exit code",
          t_main_end_to_end, needs_pct=True)
    check("no em dash in source", t_source_has_no_em_dash)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    sys.exit(1 if FAIL else 0)
