"""End-to-end flow test for the Phase-4 experiment driver, with no GPU.

    python tests/test_flow_stubbed.py

`phase3_common` only touches torch as an `@torch.no_grad()` decorator at import
time, so a tiny stub module lets it import for real. That means this test
exercises the GENUINE `merge_branches_nms` / `assemble_results` /
`config_signature` / `SurveyCheckpoint`, plus the real `p4_core.clean_control`
driver: the per-image loop, the checkpoint, the hard asserts, the ensemble
merge, the report writers and the artifact policy.

Only three things are faked, and each is stated at its definition: the model
(a fake adapter returning fixed detections/text), the defense transforms
(identity), and pycocotools' COCO eval (which is not installed here). Every
number this prints is synthetic — it is a plumbing test, not a measurement, and
it writes only into a temp directory.
"""
import io
import os
import shutil
import sys
import tempfile
import traceback
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE4 = os.path.join(REPO, "phase4")
sys.path.insert(0, REPO)
sys.path.insert(0, PHASE4)

import numpy as np                                          # noqa: E402
from PIL import Image                                       # noqa: E402

# --- stub torch BEFORE importing phase3_common -------------------------------
# phase3_common's only import-time torch use is `@torch.no_grad()`; every real
# tensor op happens inside functions this test never calls.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = lambda: (lambda fn: fn)
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False,
                                            empty_cache=lambda: None)
    sys.modules["torch"] = torch_stub

import phase3_common as pc                                  # noqa: E402
import p4_core as core                                      # noqa: E402
import p4_logging as rl                                     # noqa: E402
import p4_yolo as yolo                                      # noqa: E402
import p4_florence as flo                                   # noqa: E402

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


tmp = tempfile.mkdtemp(prefix="p4_flow_")
img_dir = os.path.join(tmp, "val2017")
os.makedirs(img_dir)
IMAGE_IDS = [139, 285, 632, 724, 776]
for i in IMAGE_IDS:                      # real 64x64 JPEGs, COCO-style names
    Image.new("RGB", (64, 48), (i % 255, 90, 140)).save(
        os.path.join(img_dir, f"{i:012d}.jpg"))
os.makedirs(os.path.join(tmp, "annotations"))
ann = os.path.join(tmp, "annotations", "instances_val2017.json")
open(ann, "w").write("{}")

# --- wire the runtime holder to real numpy/PIL and the stubbed torch ---------
core.rt.np, core.rt.Image, core.rt.pc = np, Image, pc
core.rt.torch = sys.modules["torch"]


# --- fake #1: the defense transforms. Identity, so the driver is what is -----
# under test, not the filters (those are phase3_common's and unchanged).
def fake_build_branches(pil_img, device, args, names):
    return {n: pil_img for n in names}


core.build_branches = fake_build_branches


# --- fake #2: COCO eval. pycocotools is not installed here, so this returns ---
# the 12-slot stats vector COCOeval would, with SYNTHETIC values derived from
# the condition tag (deterministic, obviously fake, never a measurement).
def fake_evaluate(args, all_results, coco_gt, image_ids):
    out = {}
    for tag in all_results:
        v = 0.30 + (len(tag) % 7) / 100.0
        out[tag] = np.array([v, v + 0.2] + [0.0] * 10)
    return out


GENUINE_EVALUATE = core.evaluate      # keep a handle before faking it
core.evaluate = fake_evaluate


# --- fake #3: the models. Fixed detections / text per condition. -------------
class FakeAdapter:
    def __init__(self, tracks, model_name="fake-model"):
        self.tracks = tuple(tracks)
        self.model_name = model_name
        self.device = "cpu"
        self.model = object()          # adaptive_bpda passes it to the attack
        self.calls = 0

    def sig_fields(self, args):
        return {"fake": True}

    def detection_fn(self, track="yolo"):
        def infer(pil_img):
            self.calls += 1
            # Two boxes, one of which overlaps across branches so the ensemble
            # NMS merge has something real to do.
            return [{"bbox": [1.0, 1.0, 10.0, 10.0], "category_id": 1,
                     "score": 0.9},
                    {"bbox": [20.0, 20.0, 8.0, 8.0], "category_id": 3,
                     "score": 0.5}]
        return infer

    def ocr_fn(self):
        def ocr(pil_img):
            self.calls += 1
            return "STOP sign ahead"
        return ocr


def make_args(entry, extra=()):
    args = entry.parse_args(["--n", "5", "--no-checkpoint",
                             "--output-dir", os.path.join(tmp, "out_" + entry.__name__),
                             *extra])
    args.image_dir, args.ann_file = img_dir, ann
    args.n = 5
    return args


print("\n[1] the real phase3_common imported and its primitives work")


def t_pc_real():
    assert hasattr(pc, "merge_branches_nms") and hasattr(pc, "assemble_results")
    # class-aware NMS: same class + IoU>0.5 collapses, different class does not
    a = [{"bbox": [0, 0, 10, 10], "category_id": 1, "score": 0.9, "image_id": 1}]
    b = [{"bbox": [1, 1, 10, 10], "category_id": 1, "score": 0.8, "image_id": 1}]
    c = [{"bbox": [0, 0, 10, 10], "category_id": 2, "score": 0.7, "image_id": 1}]
    assert len(pc.merge_branches_nms([a, b], iou_thr=0.5)) == 1, "should merge"
    assert len(pc.merge_branches_nms([a, c], iou_thr=0.5)) == 2, "different class"


check("real merge_branches_nms is class-aware at IoU 0.5", t_pc_real)


def t_assemble_clean_only():
    per_image = {1: {"clean": [], "clean+jpeg": [], "clean+median": [],
                     "clean+gaussian": []}}
    res = pc.assemble_results(per_image, defense_names=["jpeg", "median",
                                                        "gaussian"],
                              attack_tags=[],
                              ensembles={"ens_jpeg_median_gaussian":
                                         ["jpeg", "median", "gaussian"]})
    assert "clean" in res and "clean+ens_jpeg_median_gaussian" in res, res.keys()
    assert not any(t.startswith("pgd") for t in res), "no attack tags expected"


check("assemble_results with attack_tags=[] builds only clean+* conditions",
      t_assemble_clean_only)


def t_pc_call_signatures():
    """Every phase3_common call site in phase4/ must match the real signature.

    These calls are on the GPU path, so nothing else here executes them; a
    renamed kwarg in phase3_common would otherwise only surface hours into a
    real run.
    """
    import ast
    import inspect
    problems, checked = [], 0
    for f in ("p4_core.py", "p4_yolo.py", "p4_florence.py"):
        tree = ast.parse(open(os.path.join(PHASE4, f)).read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = None
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute) \
                    and fn.value.attr == "pc":
                name = fn.attr                      # rt.pc.X(...)
            elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) \
                    and fn.value.id == "pc":
                name = fn.attr                      # pc.X(...)
            if not name:
                continue
            checked += 1
            target = getattr(pc, name, None)
            assert target is not None, f"{f}: phase3_common has no '{name}'"
            if not callable(target):
                continue
            sig = inspect.signature(target)
            try:
                sig.bind(*["<pos>"] * len(node.args),
                         **{k.arg: None for k in node.keywords if k.arg})
            except TypeError as exc:
                problems.append(f"{f}: pc.{name}{sig} <- {exc}")
    assert checked >= 25, f"only found {checked} call sites — detection broke"
    assert not problems, problems


check("all phase3_common call sites match the real signatures",
      t_pc_call_signatures)


print("\n[2] clean_control runs end to end (detection track)")


def t_clean_control_detection():
    args = make_args(yolo)
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = FakeAdapter(["yolo"], "yolov8x-worldv2.pt")
    payload = core.clean_control(adapter, args, coco_gt=None,
                                 files=sorted(os.listdir(img_dir)),
                                 image_ids=IMAGE_IDS)
    assert payload["experiment"] == "clean_control"
    assert payload["num_images"] == 5
    tr = payload["tracks"]["yolo"]
    assert tr["num_images"] == 5 and tr["images_computed_this_run"] == 5
    assert tr["seconds_per_image"] is not None
    # 7 branches + 7 ensembles must all be scored
    assert len(tr["defenses"]) == 14, sorted(tr["defenses"])
    for key in ("ens_jpeg_tvm_svd", "svd", "bilateral", "blur_tvm"):
        assert key in tr["defenses"], key
    # 1 clean + 7 branch inferences per image
    assert adapter.calls == 5 * 8, adapter.calls


check("clean_control(detection) scores 7 solos + 7 ensembles on 5 images",
      t_clean_control_detection)


def t_outputs_written():
    out = os.path.join(tmp, "out_p4_yolo")
    files = sorted(os.listdir(out))
    summaries = [f for f in files if f.startswith("summary")]
    assert any(f.endswith("_latest.json") for f in summaries), files
    assert any(f.endswith("_latest.md") for f in summaries), files
    assert any(f.count("-") and f.endswith(".json") and "latest" not in f
               for f in summaries), "no timestamped summary"
    md = open(os.path.join(out, "summary_clean_control_yolo_latest.md")).read()
    for label in ("EnsJMG", "EnsJTS", "Blur+TVM", "Bilateral", "Timing"):
        assert label in md, f"markdown missing {label}"
    assert "n/a" not in md.split("## Timing")[0], "a scored cell came out n/a"
    import json
    s = json.load(open(os.path.join(out, "summary_clean_control_yolo_latest.json")))
    assert s["image_ids"] == IMAGE_IDS and s["image_ids_sha1_16"]
    assert s["published_baselines"]["yolo"] == 0.4519


check("summary json/md written, timestamped + latest, all rows present",
      t_outputs_written)


print("\n[3] clean_control runs end to end (both Florence tracks)")


def t_clean_control_florence():
    args = make_args(flo)
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = FakeAdapter(["florence_det", "florence_ocr"],
                          "microsoft/Florence-2-base")
    payload = core.clean_control(adapter, args, coco_gt=None,
                                 files=sorted(os.listdir(img_dir)),
                                 image_ids=IMAGE_IDS)
    assert set(payload["tracks"]) == {"florence_det", "florence_ocr"}
    ocr = payload["tracks"]["florence_ocr"]
    assert ocr["undefended_clean"] == 1.0
    # identical stub text => perfect self-consistency on every branch
    assert abs(ocr["clean_repeat_selfconsistency"] - 1.0) < 1e-9
    assert len(ocr["defenses"]) == 14
    assert abs(ocr["defenses"]["ens_jpeg_median_gaussian"]["value"] - 1.0) < 1e-9
    assert "OCR self-consistency" in core.TRACK_METRIC["florence_ocr"]
    # det: 1 clean + 7 branches; ocr: 2 clean + 7 branches
    assert adapter.calls == 5 * 8 + 5 * 9, adapter.calls
    md = open(os.path.join(args.output_dir,
                           "summary_clean_control_florence_latest.md")).read()
    assert "florence_det" in md and "florence_ocr" in md, "both track columns"


check("clean_control(florence) scores det + OCR from one adapter",
      t_clean_control_florence)


print("\n[4] adaptive_bpda driver runs end to end")


def run_adaptive_stubbed(extra, outdir):
    """Run yolo.adaptive_bpda with the attack math stubbed out.

    The attack MATH (BPDA surrogates, EOT, the YOLO objective, PGD) needs real
    autograd and is covered by tests/test_attack_units.py on CPU torch. Here
    both attacks return the image unchanged with a fixed, SYNTHETIC info dict
    (objective 5.0 adaptive / 8.0 oblivious, clean 10.0), so what is under
    test is everything around them: condition tags, the merge across three
    attack prefixes, the headline arithmetic and the report writers.
    """
    def fake_attack(objective):
        def attack(pil, model, device, args, *rest):
            assert isinstance(rest[-1], int), "image id must reach the attack"
            return pil, {"objective": objective, "restart": 0,
                         "restart_objectives": [objective] * args.restarts,
                         "adv_q": None}
        return attack

    saved = (yolo.check_surrogates, yolo.pgd_oblivious,
             yolo.pgd_adaptive_bpda_eot, yolo.target_objectives)
    yolo.check_surrogates = lambda *a, **k: None
    yolo.pgd_oblivious = fake_attack(8.0)
    yolo.pgd_adaptive_bpda_eot = fake_attack(5.0)
    yolo.target_objectives = lambda *a, **k: {"clean": 10.0, "oblivious": 8.0}
    try:
        args = make_args(yolo, ["--experiment", "adaptive_bpda",
                                "--output-dir", os.path.join(tmp, outdir),
                                *extra])
        os.makedirs(args.output_dir, exist_ok=True)
        adapter = FakeAdapter(["yolo"], "yolov8x-worldv2.pt")
        payload = yolo.adaptive_bpda(adapter, args, None,
                                     sorted(os.listdir(img_dir)), IMAGE_IDS)
    finally:
        (yolo.check_surrogates, yolo.pgd_oblivious,
         yolo.pgd_adaptive_bpda_eot, yolo.target_objectives) = saved
    md = open(os.path.join(args.output_dir,
                           f"summary_{yolo.summary_stem(args)}_latest.md")).read()
    return args, adapter, payload, md


def t_adaptive_driver():
    args, adapter, payload, md = run_adaptive_stubbed([], "adaptive")
    h = payload["headline"]
    for k in ("clean_mAP", "oblivious_attacked_undefended_mAP",
              "oblivious_attacked_defended_mAP",
              "adaptive_attacked_undefended_mAP",
              "adaptive_attacked_defended_mAP",
              "defense_recovery_oblivious", "defense_recovery_adaptive",
              "adaptive_minus_oblivious_defended"):
        assert k in h, f"headline missing {k}"
    # the three headline numbers must come from the same image set
    assert payload["num_images"] == 5 and payload["image_ids"] == IMAGE_IDS
    # arithmetic consistency
    assert abs(h["defense_recovery_oblivious"]
               - (h["oblivious_attacked_defended_mAP"]
                  - h["oblivious_attacked_undefended_mAP"])) < 1e-12
    assert abs(h["adaptive_minus_oblivious_defended"]
               - (h["adaptive_attacked_defended_mAP"]
                  - h["oblivious_attacked_defended_mAP"])) < 1e-12
    # every condition, including the ensemble under BOTH attacks
    for tag in ("clean", "clean+jpeg", "pgd_oblivious",
                "pgd_oblivious+ens_jpeg_median_gaussian", "pgd_adaptive",
                "pgd_adaptive+ens_jpeg_median_gaussian"):
        assert tag in payload["conditions"], f"missing condition {tag}"
    unc = payload["implementation_uncertainty"]
    assert len(unc) >= 8 and any("LOWER bound" in u for u in unc), unc
    assert payload["attack"]["alpha"] == payload["attack"]["eps"] / 4
    assert payload["defense"] == {
        "name": "EnsJMG", "key": "ens_jpeg_median_gaussian", "kind": "ensemble",
        "branches": ["jpeg", "median", "gaussian"],
        "merge": "class-aware NMS", "nms_iou": 0.5}, payload["defense"]
    # every adaptive flag, with its default, in the JSON
    assert payload["attack_flags"] == {
        "target": "ens_jpeg_median_gaussian", "eot_loss": "max", "restarts": 1,
        "median_surrogate": "true", "quantize": "truncate",
        "attack_space": "letterbox"}, payload["attack_flags"]
    assert payload["bpda"]["true_gradient_branches"] == ["median", "gaussian"]
    assert payload["bpda"]["identity_gradient_branches"] == ["jpeg"]
    so = payload["surrogate_objective"]
    assert (so["mean_clean"], so["mean_oblivious_adv"], so["mean_adaptive_adv"],
            so["frac_images_adaptive_below_oblivious"]) == (10.0, 8.0, 5.0, 1.0)
    # 12 inferences per image: (clean + 3) x (clean, oblivious, adaptive)
    assert adapter.calls == 5 * 12, adapter.calls
    for needle in ("EnsJMG", "adaptive attack, undefended",
                   "Implementation uncertainty", "LOWER bound",
                   "EOT loss `max`", "Attack space `letterbox`",
                   "quantization `truncate`", "median surrogate `true`",
                   "1 restart(s)", "| adaptive attack | EnsJMG |"):
        assert needle in md, f"markdown missing {needle!r}"
    assert os.path.isfile(os.path.join(
        args.output_dir,
        "summary_adaptive_bpda_yolo_ens_jpeg_median_gaussian_seed42_latest.json"))


check("adaptive_bpda driver: conditions, headline arithmetic, report",
      t_adaptive_driver)


def t_adaptive_driver_ens4w_flags():
    """Non-default flags reach the JSON and the markdown header."""
    args, adapter, payload, md = run_adaptive_stubbed(
        ["--target", "ens_4way", "--eot-loss", "mean", "--restarts", "2",
         "--median-surrogate", "identity", "--quantize", "round",
         "--attack-space", "native", "--seed", "7"], "adaptive_4w")
    assert payload["defense"]["name"] == "Ens4W"
    assert set(payload["defense"]["branches"]) == {"jpeg", "median", "tvm",
                                                   "gaussian"}
    assert payload["attack_flags"] == {
        "target": "ens_4way", "eot_loss": "mean", "restarts": 2,
        "median_surrogate": "identity", "quantize": "round",
        "attack_space": "native"}, payload["attack_flags"]
    a = payload["attack"]
    assert (a["restarts"], a["quantize"], a["attack_space"], a["seed"]) == \
        (2, "round", "native", 7)
    assert set(payload["bpda"]["identity_gradient_branches"]) == \
        {"jpeg", "median", "tvm"}
    for tag in ("pgd_oblivious+ens_4way", "pgd_adaptive+ens_4way",
                "pgd_adaptive+tvm"):
        assert tag in payload["conditions"], tag
    # (clean + 4 branches) x (clean, oblivious, adaptive)
    assert adapter.calls == 5 * 15, adapter.calls
    for needle in ("Ens4W", "EOT loss `mean`", "Attack space `native`",
                   "quantization `round`", "median surrogate `identity`",
                   "2 restart(s)", "seed 7", "Native space"):
        assert needle in md, f"markdown missing {needle!r}"
    assert "LOWER bound" not in md, "native space has no letterbox-scale bound"
    # ablation flags get their own stem, so the headline _latest survives
    assert os.path.isfile(os.path.join(
        args.output_dir,
        "summary_adaptive_bpda_yolo_ens_4way_seed7_eot-loss-mean_restarts-2"
        "_median-surrogate-identity_quantize-round_attack-space-native"
        "_latest.json"))


check("adaptive_bpda vs Ens4W with every non-default flag: JSON + markdown",
      t_adaptive_driver_ens4w_flags)


def t_adaptive_driver_solo():
    args, adapter, payload, md = run_adaptive_stubbed(
        ["--target", "median"], "adaptive_median")
    assert payload["defense"]["name"] == "Median"
    assert payload["defense"]["kind"] == "solo"
    assert payload["defense"]["merge"] == "none (solo transform)"
    assert not any(t.split("+")[-1].startswith("ens_")
                   for t in payload["conditions"]), "a solo target has no merge"
    h = payload["headline"]
    assert h["adaptive_attacked_defended_mAP"] == \
        payload["conditions"]["pgd_adaptive+median"]["mAP"]
    assert adapter.calls == 5 * 6, adapter.calls      # (clean + 1) x 3
    assert "Median" in md and "single branch" in md


check("adaptive_bpda vs a solo target (Median): no merge, right headline",
      t_adaptive_driver_solo)


def t_adaptive_signature_flags():
    """Every flag that changes an attacked image must change the checkpoint sig."""
    files = sorted(os.listdir(img_dir))

    def sig(extra):
        a = make_args(yolo, ["--experiment", "adaptive_bpda", *extra])
        _, branches, _ = yolo.resolve_target(a.target)
        return yolo.adaptive_signature(a, FakeAdapter(["yolo"]), a.target,
                                       branches, files, IMAGE_IDS, "cpu")

    base = sig([])
    assert base == sig([]), "signature must be deterministic"
    variants = {"target": ["--target", "ens_4way"],
                "eot_loss": ["--eot-loss", "mean"],
                "restarts": ["--restarts", "2"],
                "median_surrogate": ["--median-surrogate", "identity"],
                "quantize": ["--quantize", "round"],
                "attack_space": ["--attack-space", "native"],
                "seed": ["--seed", "7"], "eps": ["--eps", "0.1"],
                "iters": ["--pgd-iters", "50"]}
    sigs = {k: sig(v) for k, v in variants.items()}
    same = [k for k, s in sigs.items() if s == base]
    assert not same, f"flags not in the checkpoint signature: {same}"
    assert len(set(sigs.values())) == len(sigs), "two flags collided"


check("each adaptive flag (+ seed, eps, iters) changes the checkpoint signature",
      t_adaptive_signature_flags)


def t_quantize_modes():
    v = np.array([0.0, 1.4, 1.6, 254.7, 255.0], dtype=np.float32)
    assert yolo.quantize_array(v, "truncate").tolist() == [0, 1, 1, 254, 255]
    assert yolo.quantize_array(v, "round").tolist() == [0, 1, 2, 255, 255]
    assert yolo.quantize_array(v, "round").dtype == np.uint8
    try:
        yolo.quantize_array(v, "floor")
    except ValueError:
        return
    raise AssertionError("an unknown --quantize mode must raise")


check("--quantize truncate (Phase 3) vs round", t_quantize_modes)


print("\n[5] guards actually fire")


def t_missing_condition_asserts():
    """A compute_one that drops a condition must abort the run, not report."""
    args = make_args(yolo, ["--output-dir", os.path.join(tmp, "guard")])
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = pc.SurveyCheckpoint(os.path.join(args.output_dir, "c.pkl"), "sig",
                               enabled=False)
    try:
        core.run_per_image(args, "guard", sorted(os.listdir(img_dir)),
                           ["clean", "clean+jpeg"],
                           lambda pil, iid: {"clean": []},   # drops clean+jpeg
                           ckpt)
    except AssertionError as exc:
        assert "missing conditions" in str(exc), exc
        return
    raise AssertionError("a missing condition should have aborted the run")


check("run_per_image aborts when an image is missing a condition",
      t_missing_condition_asserts)


def t_failed_image_counts():
    """An image that raises must fail the count assert, not be silently dropped."""
    args = make_args(yolo, ["--output-dir", os.path.join(tmp, "guard2")])
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = pc.SurveyCheckpoint(os.path.join(args.output_dir, "c.pkl"), "sig",
                               enabled=False)

    def boom(pil, iid):
        raise RuntimeError("simulated bad image")

    try:
        core.run_per_image(args, "guard2", sorted(os.listdir(img_dir)),
                           ["clean"], boom, ckpt)
    except AssertionError as exc:
        assert "image count mismatch" in str(exc), exc
        return
    raise AssertionError("failed images should have tripped the count assert")


check("run_per_image aborts when images failed", t_failed_image_counts)


def t_ensemble_member_guard():
    saved = core.SOLO_PAPER_NAMES.copy()
    try:
        core.SOLO_PAPER_NAMES.pop("svd")          # simulate a dropped branch
        try:
            core.load_ensembles()
            core.branch_names({"ens_jpeg_tvm_svd": ["jpeg", "tvm", "svd"]})
        except AssertionError:
            return
        raise AssertionError("an unbuildable ensemble member must be caught")
    finally:
        core.SOLO_PAPER_NAMES.clear()
        core.SOLO_PAPER_NAMES.update(saved)


check("an ensemble whose member cannot be built is rejected",
      t_ensemble_member_guard)


print("\n[5b] log hygiene")


def t_quiet_eval():
    """pycocotools' boilerplate must not reach the committed run.log."""
    import contextlib
    saved_pc_eval = pc.evaluate_all_conditions

    def noisy(all_results, coco_gt, image_ids, output_dir):
        for tag in all_results:
            print("Running per image evaluation...")
            print("Evaluate annotation type *bbox*")
            print("DONE (t=1.23s).")
            for k in range(12):
                print(f" Average Precision  (AP) @[ row {k} ] = 0.123")
        return {t: np.array([0.4, 0.6] + [0.0] * 10) for t in all_results}

    pc.evaluate_all_conditions = noisy
    genuine = GENUINE_EVALUATE
    try:
        args = make_args(yolo, ["--output-dir", os.path.join(tmp, "quiet")])
        os.makedirs(args.output_dir, exist_ok=True)
        results = {"clean": [], "clean+jpeg": []}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stats = genuine(args, results, None, [1, 2])
        out = buf.getvalue()
        assert "Running per image evaluation" not in out, "boilerplate leaked"
        assert "Average Precision  (AP)" not in out, "12-row block leaked"
        assert out.count("[eval]") == 3, out          # header + 2 conditions
        assert len(stats) == 2
        # and with --verbose-eval it comes back
        args.verbose_eval = True
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            genuine(args, results, None, [1, 2])
        assert "Running per image evaluation" in buf2.getvalue()
    finally:
        pc.evaluate_all_conditions = saved_pc_eval


check("COCO-eval boilerplate is suppressed, one line per condition kept",
      t_quiet_eval)


def t_skip_clean_consistency():
    """The OCR determinism pass is skippable and actually saves a generate."""
    files = sorted(os.listdir(img_dir))
    a1 = make_args(flo, ["--output-dir", os.path.join(tmp, "occ_on")])
    os.makedirs(a1.output_dir, exist_ok=True)
    ad1 = FakeAdapter(["florence_ocr"])
    core.clean_control(ad1, a1, None, files, IMAGE_IDS)

    a2 = make_args(flo, ["--skip-clean-consistency",
                         "--output-dir", os.path.join(tmp, "occ_off")])
    os.makedirs(a2.output_dir, exist_ok=True)
    ad2 = FakeAdapter(["florence_ocr"])
    p2 = core.clean_control(ad2, a2, None, files, IMAGE_IDS)
    assert ad2.calls == ad1.calls - len(files), (ad1.calls, ad2.calls)
    assert p2["tracks"]["florence_ocr"]["clean_repeat_selfconsistency"] == 1.0


check("--skip-clean-consistency drops exactly one generate per image",
      t_skip_clean_consistency)


def t_no_markdown_duplication():
    """The full table must not be echoed into stdout (it is in the .md)."""
    import contextlib
    args = make_args(yolo, ["--output-dir", os.path.join(tmp, "nodup")])
    os.makedirs(args.output_dir, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        core.clean_control(FakeAdapter(["yolo"]), args, None,
                           sorted(os.listdir(img_dir)), IMAGE_IDS)
    out = buf.getvalue()
    md = open(os.path.join(args.output_dir,
                           "summary_clean_control_yolo_latest.md")).read()
    assert "| Defense | Kind |" not in out, "the full table was echoed to the log"
    assert "| Defense | Kind |" in md, "the table must still be in the .md"
    assert "RESULT DIGEST" in out, "the log needs a compact digest"
    assert "undefended clean" in out
    # the digest must be a fraction of the table's size
    assert len(out) < len(md) * 3, (len(out), len(md))


check("run.log carries a digest, not a copy of the markdown table",
      t_no_markdown_duplication)


print("\n[6] checkpoint resume")


def t_resume():
    out = os.path.join(tmp, "resume")
    os.makedirs(out, exist_ok=True)
    args = make_args(yolo, ["--output-dir", out])
    args.no_checkpoint = False
    adapter = FakeAdapter(["yolo"])
    files = sorted(os.listdir(img_dir))
    core.clean_control(adapter, args, None, files, IMAGE_IDS)
    first_calls = adapter.calls
    assert first_calls == 5 * 8

    # Second run, same config -> everything resumes, no inference at all.
    args2 = make_args(yolo, ["--output-dir", out])
    args2.no_checkpoint = False
    adapter2 = FakeAdapter(["yolo"])
    payload = core.clean_control(adapter2, args2, None, files, IMAGE_IDS)
    assert adapter2.calls == 0, f"resume recomputed {adapter2.calls} inferences"
    tr = payload["tracks"]["yolo"]
    assert tr["images_computed_this_run"] == 0
    assert tr["seconds_per_image"] is None, "a fully resumed run must not fake a rate"
    ckpts = [f for f in os.listdir(out) if f.startswith("checkpoint_")]
    assert len(ckpts) == 1, ckpts
    assert len(ckpts[0].split("_")[-1].split(".")[0]) == 8, "sig8 in the name"


check("a re-run resumes from checkpoint and reports no fake timing", t_resume)


print("\n[7] PIL constants used by the YOLO letterbox exist in this Pillow")


def t_pillow_constants():
    import PIL
    for name in ("BILINEAR", "BICUBIC"):
        assert hasattr(Image, name), (
            f"PIL.Image.{name} is gone in Pillow {PIL.__version__} — "
            f"p4_yolo.letterbox/unletterbox would crash")
    img = Image.new("RGB", (10, 10))
    assert img.resize((5, 5), Image.BILINEAR).size == (5, 5)
    print(f"        (Pillow {PIL.__version__})")


check("Image.BILINEAR / Image.BICUBIC still exist", t_pillow_constants)


print("\n[8] artifact policy on a real run directory")


def t_artifacts_of_a_real_run():
    rep = rl.artifact_report(os.path.join(tmp, "out_p4_yolo"))
    assert rep["committed_files"], "nothing committable was produced"
    for f in rep["committed_files"]:
        assert f.startswith("summary") or f == "run.log", f
    for f in rep["local_files"]:
        assert f.startswith("per_image") or f.startswith("checkpoint"), f
    assert rep["committed_bytes"] < 5 * 1024 ** 2, "committed output too large"


check("a real run leaves only small, committable artifacts",
      t_artifacts_of_a_real_run)


print("\n" + "=" * 60)
print(f"Phase-4 flow test: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
print("=" * 60)
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if FAIL else 0)
