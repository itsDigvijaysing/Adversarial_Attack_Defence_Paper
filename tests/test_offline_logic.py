"""Offline checks for the Phase-4 entry points. No GPU, no torch, no COCO data.

    python tests/test_offline_logic.py

The Phase-4 modules keep every heavy import inside `p4_core.load_runtime()`, so
they import cleanly here. This exercises the torch-free glue: the CLI, the
output-path scheme, seeded image selection, the OCR metric helpers, the
run-artifact policy, and the agreement between p4_core's ensemble set and the
real `phase3_common.py`. Inputs are SYNTHETIC — no number printed here is a
measurement, and nothing is written into a results directory.
"""
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE4 = os.path.join(REPO, "phase4")
sys.path.insert(0, REPO)
sys.path.insert(0, PHASE4)

import p4_core as core              # noqa: E402
import p4_florence as flo           # noqa: E402
import p4_logging as rl             # noqa: E402
import p4_yolo as yolo              # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception:
        FAIL.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()


tmp = tempfile.mkdtemp(prefix="p4_offline_check_")
img_dir = os.path.join(tmp, "val2017")
os.makedirs(img_dir)
for i in range(1, 21):
    open(os.path.join(img_dir, f"{i:012d}.jpg"), "wb").close()
os.makedirs(os.path.join(tmp, "annotations"))
ann = os.path.join(tmp, "annotations", "instances_val2017.json")
open(ann, "w").write("{}")

DEFENSE_DEFAULTS = (75, 3, 0.05, 200, 1.0, 0.90, 0.5)


def defense_tuple(a):
    return (a.jpeg_quality, a.median_kernel, a.tvm_weight, a.tvm_iters,
            a.gaussian_sigma, a.svd_keep_ratio, a.nms_iou)


print("\n[1] CLI — both entry points, Phase-3 defaults")


def t_yolo_defaults():
    a = yolo.parse_args([])
    assert a.experiment == "clean_control", "first experiment is the default"
    assert a.n == 1000 and a.select == "head" and a.seed == 42
    assert defense_tuple(a) == DEFENSE_DEFAULTS, defense_tuple(a)
    assert a.eps == 0.03 and a.pgd_iters == 10
    assert abs(a.alpha - 0.03 / 4) < 1e-12, "alpha must default to eps/4"
    assert abs(yolo.parse_args(["--eps", "0.08"]).alpha - 0.02) < 1e-12


check("p4_yolo defaults match Phase 3 (defenses, eps, alpha=eps/4)",
      t_yolo_defaults)


def t_florence_defaults():
    a = flo.parse_args([])
    assert a.experiment == "clean_control"
    assert a.tracks == list(flo.ALL_TRACKS), "both tracks from one model load"
    assert defense_tuple(a) == DEFENSE_DEFAULTS, defense_tuple(a)
    assert a.num_beams == 5 and a.det_max_new_tokens == 512
    assert a.ocr_max_new_tokens == 256 and a.repetition_penalty == 1.8
    assert flo.parse_args(["--tracks", "florence_ocr"]).tracks == ["florence_ocr"]


check("p4_florence defaults match Phase 3 (both tracks, 5 beams, 512 tokens)",
      t_florence_defaults)


def t_alias():
    assert yolo.parse_args(["--n", "7"]).n == 7
    assert yolo.parse_args(["--num-images", "7"]).n == 7
    assert flo.parse_args(["--num-images", "7"]).n == 7


check("--n / --num-images alias on both entry points", t_alias)


def t_experiments_exposed():
    assert yolo.EXPERIMENTS == ("clean_control", "adaptive_bpda")
    assert flo.EXPERIMENTS == ("clean_control",)
    for bad in (["--experiment", "adaptive_bpda"],):
        try:
            flo.parse_args(bad)
        except SystemExit:
            break
    else:
        raise AssertionError("Florence must reject the YOLO-only experiment")


check("each model exposes only the experiments it implements",
      t_experiments_exposed)


def t_smoke_overrides():
    a = yolo.parse_args(["--smoke-test", "--experiment", "adaptive_bpda"])
    assert a.n == 2 and a.device == "cpu" and a.no_checkpoint
    assert a.output_dir.endswith("smoke_test")
    assert a.pgd_iters == 2, "smoke test shortens PGD"
    b = yolo.parse_args(["--smoke-test", "--pgd-iters", "10"])
    assert b.pgd_iters == 10, "an explicit --pgd-iters wins"
    c = flo.parse_args(["--smoke-test"])
    assert c.n == 2 and c.device == "cpu" and c.num_beams == 1
    # an explicit --device must survive --smoke-test, else the GPU path can
    # never be smoke-tested at all
    g = flo.parse_args(["--smoke-test", "--device", "cuda"])
    assert g.device == "cuda" and g.n == 2, vars(g)
    assert yolo.parse_args(["--smoke-test", "--device", "cuda"]).device == "cuda"
    assert yolo.parse_args([]).device == "auto"
    d = flo.parse_args(["--smoke-test", "--full-gen"])
    assert d.num_beams == 5, "--full-gen restores the Phase-3 decode"


check("--smoke-test overrides (2 images, CPU, isolated dir, cheap decode)",
      t_smoke_overrides)


print("\n[2] output-path scheme (stable dir + timestamped summaries)")


def t_output_paths():
    a = yolo.parse_args(["--experiment", "adaptive_bpda"])
    assert a.output_dir == os.path.join(core.DEFAULT_RESULTS_ROOT,
                                       "adaptive_bpda_yolo"), a.output_dir
    b = flo.parse_args([])
    assert b.output_dir == os.path.join(core.DEFAULT_RESULTS_ROOT,
                                       "clean_control_florence"), b.output_dir
    assert len(a.run_id) == 15 and a.run_id[8] == "-", a.run_id
    c = yolo.parse_args(["--output-dir", "/tmp/x"])
    assert c.output_dir == "/tmp/x", "explicit --output-dir must win"


check("output dir is <results-root>/<experiment>_<model>, run_id stamped",
      t_output_paths)


def t_summary_naming():
    a = yolo.parse_args(["--n", "300", "--output-dir",
                         os.path.join(tmp, "outdir")])
    os.makedirs(a.output_dir, exist_ok=True)
    stamped, latest = core.write_summary(a, {"clean_mAP": 0.0}, "demo_yolo")
    md_stamped, md_latest = core.write_markdown(a, "# demo\n", "demo_yolo")
    for p in (stamped, latest, md_stamped, md_latest):
        assert os.path.isfile(p), p
        assert os.path.basename(p).startswith("summary"), (
            "every summary file must start with 'summary' or .gitignore's "
            "!results/**/summary*.json negation will not rescue it")
    assert a.run_id in os.path.basename(stamped), "timestamp missing"
    assert "n300" in os.path.basename(stamped), "image count missing"
    assert a.run_id not in os.path.basename(latest), "_latest must be stable"
    import json
    assert json.load(open(stamped))["run_id"] == a.run_id


check("summaries are timestamped + mirrored to _latest, all 'summary*'",
      t_summary_naming)


def t_no_overwrite():
    """A second run in the same directory must not clobber the first."""
    a = yolo.parse_args(["--output-dir", os.path.join(tmp, "outdir2")])
    os.makedirs(a.output_dir, exist_ok=True)
    first, _ = core.write_summary(a, {"v": 1}, "demo_yolo")
    a.run_id = "20260817-999999"          # simulate a later run
    second, _ = core.write_summary(a, {"v": 2}, "demo_yolo")
    assert first != second, "two runs collided on one filename"
    import json
    assert json.load(open(first))["v"] == 1, "the earlier run was overwritten"


check("a later run never overwrites an earlier summary", t_no_overwrite)


print("\n[3] path resolution")
cwd = os.getcwd()
os.chdir(tmp)
try:
    check("resolve_image_dir finds ./val2017",
          lambda: core.resolve_image_dir(None).endswith("val2017") or _raise())
    check("resolve_ann_file finds ./annotations/...",
          lambda: core.resolve_ann_file(None).endswith(
              "instances_val2017.json") or _raise())
finally:
    os.chdir(cwd)


def _raise():
    raise AssertionError("resolution returned an unexpected path")


print("\n[4] seeded, reproducible image selection")


def t_head():
    f1, i1 = core.select_images(img_dir, 5, 42, "head")
    f2, i2 = core.select_images(img_dir, 5, 999, "head")
    assert i1 == i2 == [1, 2, 3, 4, 5], (i1, i2)
    assert f1 == f2, "head selection must ignore the seed"


check("head selection = first n of sorted list, seed-independent", t_head)


def t_random_repeatable():
    a = core.select_images(img_dir, 5, 42, "random")[1]
    b = core.select_images(img_dir, 5, 42, "random")[1]
    c = core.select_images(img_dir, 5, 43, "random")[1]
    assert a == b, "same seed must give the same subset"
    assert a != c, "different seed should give a different subset"
    assert a == sorted(a) and len(set(a)) == 5


check("random selection is seeded + reproducible", t_random_repeatable)


def t_digest():
    d1, d2 = core.ids_digest([3, 1, 2]), core.ids_digest([1, 2, 3])
    assert d1 == d2 and len(d1) == 16, "digest must be order-independent"
    assert core.ids_digest([1, 2, 4]) != d1


check("ids_digest is stable and order-independent", t_digest)


def t_selection_guard():
    bad = os.path.join(tmp, "bad")
    os.makedirs(bad, exist_ok=True)
    open(os.path.join(bad, "not_a_coco_name.jpg"), "wb").close()
    try:
        core.select_images(bad, 1, 42, "head")
    except AssertionError:
        return
    raise AssertionError("non-numeric filename should have been rejected")


check("non-COCO filenames are rejected (no silent bad image_id)",
      t_selection_guard)


def t_short_dir_is_an_error():
    """--n larger than the image dir must stop the run (20 images here)."""
    for mode in ("head", "random"):
        try:
            core.select_images(img_dir, 50, 42, mode)
        except RuntimeError as exc:
            msg = str(exc)
            assert "n=50" in msg and "only 20" in msg, msg
            assert "val2017" in msg and "missing" in msg, msg
            assert "http://images.cocodataset.org/zips/val2017.zip" in msg, msg
        else:
            raise AssertionError(f"{mode}: a short image dir must raise")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):       # --smoke-test path: warn only
        files, ids = core.select_images(img_dir, 50, 42, "head",
                                        allow_short=True)
    assert len(files) == len(ids) == 20 and "[WARN]" in buf.getvalue()
    assert core.select_images(img_dir, 20, 42, "head")[1] == list(range(1, 21))


check("fewer images than --n is an error naming val2017.zip (smoke: warning)",
      t_short_dir_is_an_error)


def t_preflight_counts_images():
    """--preflight says READY only when the run itself can start."""
    import contextlib
    import io

    def run(img, *flags):
        a = yolo.parse_args(["--image-dir", img, "--ann-file", ann] + list(flags))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = core.preflight(a, modules=[], weights=[])
        return rc, buf.getvalue()

    rc, out = run(img_dir, "--n", "50")              # 20 on disk < 50
    assert rc == 1 and "SHORT" in out and "holds 20 images" in out, out
    assert core.COCO_VAL2017_URL in out and "NOT READY" in out, out
    rc, out = run(img_dir, "--n", "20")
    assert rc == 0 and "(20 images)" in out and "[verdict] READY" in out, out
    one = os.path.join(tmp, "one_image")
    os.makedirs(one, exist_ok=True)
    open(os.path.join(one, "000000000001.jpg"), "wb").close()
    rc, out = run(one, "--smoke-test")               # smoke: n=2, 1 is allowed
    assert rc == 0 and "(1 images)" in out, out
    empty = os.path.join(tmp, "no_images")
    os.makedirs(empty, exist_ok=True)
    rc, out = run(empty, "--smoke-test")             # select_images would raise
    assert rc == 1 and "holds 0 images" in out, out


check("preflight: READY only if the image dir holds --n images (smoke: >= 1)",
      t_preflight_counts_images)


print("\n[5] OCR metric helpers (verbatim Phase-3 copies)")


def t_similarity():
    assert core.text_similarity("STOP", "STOP") == 1.0
    assert core.text_similarity("STOP", "stop") == 1.0, "normalize lowercases"
    assert 0.0 <= core.text_similarity("STOP", "GO") < 1.0
    # Faithful to the Phase-3 helper: normalize_text() strips BEFORE the tag
    # regex, so tag removal leaves padding spaces. Symmetric between the two
    # OCR outputs compared (both come from the same decode), hence 1.0 here.
    assert core.text_similarity("<s>hello</s>", "<t>hello</t>") == 1.0
    assert core.normalize_text("<s>hello</s>") == " hello "


check("text_similarity: self=1.0, tag/case normalisation", t_similarity)


def t_vote():
    assert core.consensus_vote_text(["a", "a", "b"]) == "a", "majority wins"
    assert core.consensus_vote_text([]) == ""
    assert core.consensus_vote_text(["only"]) == "only"


check("consensus_vote_text majority + edge cases", t_vote)


def t_decode():
    assert core.decode_ocr({"<OCR>": "text"}) == "text"
    assert core.decode_ocr({"<OCR>": ["a", "b"]}) == "a b"
    assert core.decode_ocr({"<OCR>": {"text": "x"}}) == "x"


check("decode_ocr handles str/list/dict payloads", t_decode)


print("\n[6] defense set agrees with the real phase3_common.py")


def _parse_phase3_dicts():
    """Evaluate phase3_common's ensemble constants WITHOUT importing torch.

    phase3_common.py imports torch at module scope, so it cannot be imported
    here. This walks its AST and evaluates only the list/dict literals (handling
    `**other` unpacking and `a + b` list concatenation), which is enough to read
    SOLO_DEFENSES / *_ENSEMBLES exactly as the real module would build them.
    """
    src = open(os.path.join(REPO, "phase3_common.py")).read()
    ns = {}

    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return ns[node.id]
        if isinstance(node, ast.List):
            return [ev(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            out = {}
            for k, v in zip(node.keys, node.values):
                if k is None:                      # **other
                    out.update(ev(v))
                else:
                    out[ev(k)] = ev(v)
            return out
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return ev(node.left) + ev(node.right)
        raise ValueError(ast.dump(node))

    for stmt in ast.parse(src).body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and \
                isinstance(stmt.targets[0], ast.Name):
            try:
                ns[stmt.targets[0].id] = ev(stmt.value)
            except Exception:                      # not a literal we care about
                continue
    return ns


def t_ensembles_match_phase3():
    ns = _parse_phase3_dicts()
    survey = ns["SURVEY_ENSEMBLES"]
    for key, label in core.ENSEMBLE_PAPER_NAMES.items():
        assert key in survey, (
            f"{label} ({key}) is not in phase3_common.SURVEY_ENSEMBLES — "
            f"Phase 4 would score a different ensemble than Phase 3")
    branches = core.branch_names({k: survey[k]
                                  for k in core.ENSEMBLE_PAPER_NAMES})
    assert set(branches) == {"jpeg", "median", "gaussian", "tvm", "blur_tvm",
                             "bilateral", "svd"}, branches
    # Every member of every scored ensemble must be a branch we actually build.
    for key in core.ENSEMBLE_PAPER_NAMES:
        for m in survey[key]:
            assert m in branches, f"{key} member {m} is not built"


check("all 7 paper ensembles exist in phase3_common with buildable members",
      t_ensembles_match_phase3)


def t_bpda_targets_from_phase3():
    """--target reads its branch lists from phase3_common, never restates them."""
    ns = _parse_phase3_dicts()
    tier1 = ns["TIER1_ENSEMBLES"]
    assert yolo.DEFAULT_TARGET == "ens_jpeg_median_gaussian"
    expected = {"ens_jpeg_median_gaussian": ("EnsJMG",
                                             {"jpeg", "median", "gaussian"}),
                "ens_4way": ("Ens4W", {"jpeg", "median", "tvm", "gaussian"}),
                "ens_jpeg_median_tvm": ("EnsJMT", {"jpeg", "median", "tvm"})}
    for key, (label, members) in expected.items():
        assert key in yolo.ENSEMBLE_TARGETS, key
        ens, branches, paper = yolo.resolve_target(key, table=tier1)
        assert branches == tier1[key] and set(branches) == members, branches
        assert ens == {key: tier1[key]} and paper == label, (ens, paper)
    for key in yolo.ENSEMBLE_TARGETS:     # every choice resolves + has surrogates
        assert yolo.resolve_target(key, table=tier1)[1] == tier1[key]
    for solo, label in (("median", "Median"), ("jpeg", "JPEG"),
                        ("gaussian", "Gaussian"), ("tvm", "TVM")):
        assert yolo.resolve_target(solo, table=tier1) == ({}, [solo], label)
    try:                                   # svd has no surrogate -> not a target
        yolo.resolve_target("ens_jpeg_tvm_svd", table=tier1)
    except ValueError:
        pass
    else:
        raise AssertionError("EnsJTS (svd member) must not be a --target")
    # No module constant, and nothing in the target-resolving or driver code,
    # may restate an ensemble's members as a literal. (bpda_branch's
    # identity-surrogate tuple is a set of BRANCH TYPES that merely coincides
    # with EnsJMT, so the scan is scoped instead of file-wide.)
    tree = ast.parse(open(os.path.join(PHASE4, "p4_yolo.py")).read())
    scopes = [s for s in tree.body if isinstance(s, (ast.Assign, ast.AnnAssign))]
    scopes += [s for s in tree.body if isinstance(s, ast.FunctionDef)
               and s.name in ("resolve_target", "adaptive_bpda", "parse_args")]
    assert len(scopes) > 10, "scope detection broke"
    member_sets = [set(v) for v in tier1.values()]
    for scope in scopes:
        for node in ast.walk(scope):
            if isinstance(node, (ast.List, ast.Tuple)) and node.elts and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str)
                    for e in node.elts):
                assert {e.value for e in node.elts} not in member_sets, (
                    f"p4_yolo.py:{node.lineno} hard-codes an ensemble's members")


check("--target ensembles come from phase3_common (EnsJMG/Ens4W/EnsJMT + solos)",
      t_bpda_targets_from_phase3)


def t_adaptive_flags():
    a = yolo.parse_args(["--experiment", "adaptive_bpda"])
    assert (a.target, a.eot_loss, a.restarts, a.median_surrogate, a.quantize,
            a.attack_space) == ("ens_jpeg_median_gaussian", "max", 1, "true",
                                "truncate", "letterbox"), vars(a)
    b = yolo.parse_args(["--target", "ens_4way", "--eot-loss", "mean",
                         "--restarts", "3", "--median-surrogate", "identity",
                         "--quantize", "round", "--attack-space", "native"])
    assert (b.target, b.eot_loss, b.restarts, b.median_surrogate, b.quantize,
            b.attack_space) == ("ens_4way", "mean", 3, "identity", "round",
                                "native")
    assert tuple(yolo.ATTACK_FLAGS) == ("target", "eot_loss", "restarts",
                                        "median_surrogate", "quantize",
                                        "attack_space")
    import contextlib
    import io
    for bad in (["--restarts", "0"], ["--target", "ens_jpeg_tvm_svd"],
                ["--eot-loss", "sum"], ["--quantize", "floor"],
                ["--attack-space", "square"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                yolo.parse_args(bad)
            except SystemExit:
                continue
        raise AssertionError(f"{bad} should have been rejected")


check("adaptive flags: defaults (EnsJMG, max, R=1, true median, truncate, "
      "letterbox) and choices", t_adaptive_flags)


def t_pgd_init_seed():
    s = yolo.pgd_init_seed
    assert s(42, 139, 0) == s(42, 139, 0), "must be deterministic"
    seen = {s(seed, iid, r) for seed in (0, 1, 42) for iid in (139, 285)
            for r in (0, 1, 2)}
    assert len(seen) == 18, "seed, image id and restart must all change the draw"
    assert all(0 <= v < 2 ** 63 for v in seen), "torch manual_seed range"
    # restart r keeps its draw whatever --restarts is: no dependence on R
    assert s(42, 139, 1) == yolo.pgd_init_seed(42, 139, restart=1)


check("PGD init seed: deterministic per (seed, image id, restart)",
      t_pgd_init_seed)


def t_summary_stem():
    a = yolo.parse_args(["--experiment", "adaptive_bpda", "--seed", "1"])
    b = yolo.parse_args(["--experiment", "adaptive_bpda", "--seed", "2"])
    c = yolo.parse_args(["--experiment", "adaptive_bpda", "--target", "median"])
    stems = {yolo.summary_stem(x) for x in (a, b, c)}
    assert len(stems) == 3, "each (target, seed) needs its own _latest"
    assert yolo.summary_stem(a) == \
        "adaptive_bpda_yolo_ens_jpeg_median_gaussian_seed1"
    # an ablation in the same dir must not take over the headline's _latest
    assert set(yolo.STEM_FLAG_DEFAULTS) == set(yolo.ATTACK_FLAGS) - {"target"}
    assert yolo.STEM_FLAG_DEFAULTS == {
        k: getattr(a, k) for k in yolo.STEM_FLAG_DEFAULTS}, "stale defaults"
    m = yolo.parse_args(["--experiment", "adaptive_bpda", "--seed", "1",
                         "--median-surrogate", "identity"])
    assert yolo.summary_stem(m) == yolo.summary_stem(a) + \
        "_median-surrogate-identity"
    for flag, val in (("--eot-loss", "mean"), ("--restarts", "2"),
                      ("--quantize", "round"), ("--attack-space", "native")):
        x = yolo.parse_args(["--experiment", "adaptive_bpda", "--seed", "1",
                             flag, val])
        assert yolo.summary_stem(x) == \
            f"{yolo.summary_stem(a)}_{flag[2:]}-{val}", flag
    assert yolo.condition_label("pgd_adaptive+ens_4way") == \
        ("adaptive attack", "Ens4W")
    assert yolo.condition_label("pgd_oblivious") == \
        ("oblivious attack", "undefended")
    assert yolo.condition_label("clean+median") == ("clean", "Median")


check("adaptive summaries: one stem per (target, seed, ablation flags); "
      "paper labels",
      t_summary_stem)


def t_uncertainty_text():
    base = yolo.parse_args(["--experiment", "adaptive_bpda"])
    jmg = ["jpeg", "median", "gaussian"]
    lb = " ".join(yolo.implementation_uncertainty(base, "EnsJMG", jmg))
    for needle in ("Scale mismatch", "LOWER bound", "Geometry mismatch",
                   "bicubic", "truncation", "union", "jpeg"):
        assert needle in lb, f"letterbox text lacks {needle!r}"
    assert "a.e. gradient" in lb, "median is a TRUE gradient by default"
    nat = yolo.parse_args(["--experiment", "adaptive_bpda", "--attack-space",
                           "native", "--median-surrogate", "identity",
                           "--quantize", "round", "--eot-loss", "mean",
                           "--restarts", "3"])
    nt = " ".join(yolo.implementation_uncertainty(nat, "EnsJMG", jmg))
    assert "Native space" in nt and "Scale mismatch" not in nt
    assert "interpolation" in nt, "native must still state its residual mismatch"
    assert "identity Jacobian (straight-through) for jpeg, median" in nt
    assert "rounded to uint8" in nt and "3 restarts" in nt
    assert "clipped to the clean image +/- 7/255" in nt, "native eps clip"
    assert "8/255 (eps*255 = 7.65)" in lb, "letterbox must state its overshoot"
    assert (yolo.uint8_budget(0.03), yolo.uint8_budget(0.06),
            yolo.uint8_budget(8 / 255), yolo.uint8_budget(0.1)) == \
        (7, 15, 8, 25)
    assert "'mean'" in nt
    solo = " ".join(yolo.implementation_uncertainty(base, "Median", ["median"]))
    assert "union" not in solo, "a solo target has no union merge"


check("implementation_uncertainty tracks every attack option",
      t_uncertainty_text)


def t_surrogate_table():
    """Every surrogate helper reads SURROGATES: one classification, no drift."""
    assert yolo.SURROGATE_BRANCHES == tuple(yolo.SURROGATES)
    expect = {"true": {"jpeg", "tvm"}, "identity": {"jpeg", "tvm", "median"}}
    for mode, ident in expect.items():
        a = yolo.parse_args(["--experiment", "adaptive_bpda",
                             "--median-surrogate", mode])
        got = {b for b in yolo.SURROGATE_BRANCHES
               if yolo.is_identity_surrogate(b, a)}
        assert got == ident, (mode, got)
        for b in yolo.SURROGATE_BRANCHES:
            kind, text = yolo.surrogate_spec(b, a)
            assert kind in ("identity", "gaussian", "median"), (b, kind)
            assert yolo.surrogate_description(b, a) == text
            # the identity flag comes from the kind, never from the wording
            assert (kind == "identity") == yolo.is_identity_surrogate(b, a)
    try:
        yolo.surrogate_spec("svd", a)
    except ValueError:
        return
    raise AssertionError("a branch without a surrogate must raise")


check("BPDA surrogates: one table drives kind, text and the identity split",
      t_surrogate_table)


def t_unknown_member_rejected():
    try:
        core.branch_names({"bogus": ["nlm"]})
    except AssertionError:
        return
    raise AssertionError("an unknown ensemble member should be rejected")


check("branch_names rejects an ensemble member it cannot build",
      t_unknown_member_rejected)


print("\n[7] run-artifact policy (p4_logging.py)")


def t_run_log_tee():
    log = os.path.join(tmp, "logs", "run.log")
    with rl.run_log(log, argv=["prog", "--n", "3"], extra={"n": 3}):
        print("hello from inside the run")
    body = open(log).read()
    for needed in ("RUN START", "hello from inside the run", "RUN END",
                   "prog --n 3", "n       : 3", "git", "OK"):
        assert needed in body, f"run.log missing {needed!r}"
    with rl.run_log(log, argv=["prog", "--n", "4"]):
        print("second session")
    body2 = open(log).read()
    assert body2.count("RUN START") == 2, "run.log must append, not truncate"
    assert "hello from inside the run" in body2, "first session was destroyed"


check("run_log tees to a stamped, appending run.log", t_run_log_tee)


def t_run_log_records_failure():
    log = os.path.join(tmp, "logs", "fail.log")
    try:
        with rl.run_log(log, argv=["prog"]):
            raise ValueError("boom")
    except ValueError:
        pass
    else:
        raise AssertionError("run_log must re-raise, not swallow")
    assert "FAILED (ValueError: boom)" in open(log).read()


check("run_log records a crash and re-raises it", t_run_log_records_failure)


def t_eval_dumps_are_ephemeral():
    out = os.path.join(tmp, "dumps_off")
    os.makedirs(out, exist_ok=True)
    with rl.eval_dump_dir(out, keep=False) as d:
        assert os.path.isdir(d)
        assert not d.startswith(out), "dumps must NOT be inside the run dir"
        open(os.path.join(d, "clean_jpeg.json"), "w").write("[]")
        held = d
    assert not os.path.exists(held), "temp dump dir must be deleted"
    assert os.listdir(out) == [], "run dir must stay clean of dumps"


check("eval dumps go to a temp dir and are deleted", t_eval_dumps_are_ephemeral)


def t_eval_dumps_keep():
    out = os.path.join(tmp, "dumps_on")
    os.makedirs(out, exist_ok=True)
    with rl.eval_dump_dir(out, keep=True) as d:
        assert d == os.path.join(out, "eval_dumps")
        open(os.path.join(d, "clean_jpeg.json"), "w").write("[]")
    assert os.path.isfile(os.path.join(out, "eval_dumps", "clean_jpeg.json"))


check("--keep-eval-dumps keeps them under the run dir", t_eval_dumps_keep)


def t_artifact_classification():
    out = os.path.join(tmp, "artifacts")
    os.makedirs(out, exist_ok=True)
    for name in ("summary_clean_control_yolo_n1000_20260817-101010.json",
                 "summary_clean_control_yolo_latest.json",
                 "summary_clean_control_yolo_latest.md", "run.log",
                 "per_image_yolo.json", "checkpoint_yolo_abcd1234.pkl",
                 "clean_jpeg.json"):
        open(os.path.join(out, name), "wb").write(b"x" * 100)
    rep = rl.artifact_report(out)
    assert len(rep["committed_files"]) == 4, rep["committed_files"]
    assert set(rep["local_files"]) == {"per_image_yolo.json",
                                       "checkpoint_yolo_abcd1234.pkl",
                                       "clean_jpeg.json"}, rep["local_files"]


check("artifact_report splits committed vs local", t_artifact_classification)


def t_gitignore_agreement():
    """The classifier must agree with the repo's real .gitignore rules."""
    out = os.path.join(REPO, "results", "phase4", "_policy_probe")
    os.makedirs(out, exist_ok=True)
    try:
        for name in ("summary_clean_control_yolo_n1000_20260817-101010.json",
                     "summary_clean_control_yolo_latest.json",
                     "notes.md", "run.log", "per_image_yolo.json",
                     "checkpoint_yolo_abcd1234.pkl", "clean_jpeg.json"):
            path = os.path.join(out, name)
            open(path, "w").close()
            ignored = subprocess.run(["git", "check-ignore", "-q", path],
                                     cwd=REPO).returncode == 0
            assert rl.is_committed(name) == (not ignored), (
                f"{name}: p4_logging says committed={rl.is_committed(name)} "
                f"but git says ignored={ignored}")
    finally:
        shutil.rmtree(out, ignore_errors=True)


check("p4_logging's committed/local split matches git check-ignore",
      t_gitignore_agreement)


def t_smoke_output_is_ignored():
    """Smoke-test output must be impossible to commit."""
    a = yolo.parse_args(["--smoke-test"])
    probe = os.path.join(a.output_dir, "summary_clean_control_yolo_latest.json")
    os.makedirs(os.path.dirname(probe), exist_ok=True)
    open(probe, "w").close()
    try:
        ignored = subprocess.run(["git", "check-ignore", "-q", probe],
                                 cwd=REPO).returncode == 0
        assert ignored, "smoke-test summaries must be gitignored"
    finally:
        shutil.rmtree(a.output_dir, ignore_errors=True)


check("smoke-test output is gitignored", t_smoke_output_is_ignored)


def t_new_flags():
    for a in (yolo.parse_args([]), flo.parse_args([])):
        assert a.keep_eval_dumps is False and a.no_per_image is False
    assert yolo.parse_args(["--keep-eval-dumps"]).keep_eval_dumps is True


check("--keep-eval-dumps / --no-per-image exist and default off", t_new_flags)


print("\n[8] report helpers")


def t_timing_table():
    results = {
        "florence_det": {"num_images": 100, "images_computed_this_run": 100,
                         "seconds": 400.0, "seconds_per_image": 4.0},
        "florence_ocr": {"num_images": 100, "images_computed_this_run": 0,
                         "seconds": 1.0, "seconds_per_image": None},
    }
    md = "\n".join(core.timing_table(results, "cuda:0"))
    assert "florence_det" in md and "4.00" in md
    assert "n/a (all resumed)" in md, "a fully resumed track must not fake a rate"
    assert "est. n=5000" in md
    core.print_timing(results, "cuda:0")


check("timing_table reports a real rate and flags resumed tracks",
      t_timing_table)


def t_header_lines():
    a = yolo.parse_args(["--n", "5"])
    a.image_dir, a.ann_file = img_dir, ann
    md = "\n".join(core.header_lines(a, [1, 2, 3, 4, 5]))
    assert "id digest" in md and a.run_id in md and "yolo" in md
    assert "SMOKE" not in md, "a real run must not carry the smoke banner"
    b = yolo.parse_args(["--smoke-test"])
    b.image_dir, b.ann_file = img_dir, ann
    smoke_md = "\n".join(core.header_lines(b, [1, 2]))
    assert "NOT A RESULT" in smoke_md, "smoke output must self-label"
    assert "2 images, device `cpu`" in smoke_md, smoke_md
    # the banner states the REAL n and device, not a hard-coded "2 images, CPU"
    c = yolo.parse_args(["--smoke-test", "--device", "cuda"])
    c.image_dir, c.ann_file = img_dir, ann
    gpu_md = "\n".join(core.header_lines(c, [1, 2, 3], device="cuda:0"))
    assert "3 images, device `cuda:0`" in gpu_md, gpu_md


check("header_lines records provenance and labels smoke runs", t_header_lines)


print("\n" + "=" * 60)
print(f"Phase-4 offline check: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
print("=" * 60)
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if FAIL else 0)
