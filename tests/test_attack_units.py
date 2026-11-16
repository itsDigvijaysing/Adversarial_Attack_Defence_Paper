"""CPU unit tests for the Phase-4 adaptive-attack math in p4_yolo.py.

    CUDA_VISIBLE_DEVICES='' /home/king/miniconda3/envs/vlm_ftune/bin/python \
        tests/test_attack_units.py

Needs torch (the vlm_ftune env); under a Python without torch it prints SKIP
and exits 0, so it is safe in any test loop. It forces CPU itself
(CUDA_VISIBLE_DEVICES='' before torch is imported).

No model weights, no COCO, no network. The detector is a FAKE: a small random
conv whose output has YOLO's [1, 4+80, A] layout with post-sigmoid class rows,
so the objective, EOT, BPDA and PGD code paths run for real while every
number they produce is synthetic. The defense transforms are the REAL
phase3_common ones on small random images. The native-letterbox check imports
ultralytics' LetterBox class only (a pure numpy/cv2 transform; YOLO_OFFLINE=1
stops its import-time DNS probe) and is skipped if ultralytics is missing.
"""
import os
import sys
import traceback
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only, always
os.environ.setdefault("YOLO_OFFLINE", "1")       # no DNS probe from ultralytics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE4 = os.path.join(REPO, "phase4")
sys.path.insert(0, REPO)
sys.path.insert(0, PHASE4)

try:
    import torch
except ImportError:
    print("SKIP tests/test_attack_units.py: torch is not installed in this "
          "Python. Run it with the vlm_ftune env.")
    sys.exit(0)

import numpy as np                                          # noqa: E402
from PIL import Image                                       # noqa: E402

import phase3_common as pc                                  # noqa: E402
import p4_core as core                                      # noqa: E402
import p4_yolo as yolo                                      # noqa: E402

core.rt.np, core.rt.torch, core.rt.Image, core.rt.pc = np, torch, Image, pc
F = torch.nn.functional
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "this test must run on CPU only"

PASS, FAIL, SKIP = [], [], []


class Skip(Exception):
    pass


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Skip as exc:
        SKIP.append(name)
        print(f"  SKIP  {name} ({exc})")
    except Exception:
        FAIL.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()


def rand_pil(w, h, seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def make_args(*extra):
    return yolo.parse_args(["--experiment", "adaptive_bpda",
                            "--yolo-imgsz", "32", *extra])


# --- the fake detector ---------------------------------------------------------
class _Head(torch.nn.Module):
    """yolo_anchor_scores resets `.shape` on the last layer before a forward."""
    shape = None


class _FakeDetector(torch.nn.Module):
    """[1,3,H,W] -> ([1, 4+80, H*W/16], None), class rows post-sigmoid."""

    def __init__(self, ncls=80):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 4 + ncls, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.randn(self.conv.weight.shape, generator=g) * 0.6)
            self.conv.bias.copy_(torch.randn(self.conv.bias.shape, generator=g))
        self.model = torch.nn.ModuleList([_Head()])
        self.stride = torch.tensor([8.0, 16.0, 32.0])

    def forward(self, x):
        y = F.avg_pool2d(self.conv(x), 4).flatten(2)
        return (torch.cat([y[:, :4], y[:, 4:].sigmoid()], dim=1), None)


class FakeYolo:
    def __init__(self):
        self.model = _FakeDetector().eval()


MODEL = FakeYolo()

print("\n[1] BPDA surrogates: forward = the deployed transform, real gradient")


def t_median_forward():
    for shape, k in (((2, 3, 37, 41), 3), ((1, 3, 20, 23), 5)):
        x = torch.rand(shape, generator=torch.Generator().manual_seed(1))
        diff = (yolo.median_differentiable(x, k)
                - pc.median_gpu(x, k)).abs().max().item()
        assert diff < 1e-7, f"k={k}: |diff| = {diff}"
    yolo.assert_median_matches(CPU, 3)          # the runtime guard agrees


check("median_differentiable == pc.median_gpu (k=3, k=5, reflect pad)",
      t_median_forward)


def t_median_gradient():
    x = torch.rand(1, 3, 24, 29, generator=torch.Generator().manual_seed(2),
                   dtype=torch.float64).requires_grad_(True)
    y = yolo.median_differentiable(x, 3)
    (g,) = torch.autograd.grad(y.sum(), x, retain_graph=True)
    assert g.abs().sum() > 0, "median gradient is all zero"
    # each output routes exactly 1 to the input pixel its window selected
    assert abs(g.sum().item() - y.numel()) < 1e-9, (g.sum().item(), y.numel())
    assert torch.equal(g, g.round()) and g.min() >= 0, "counts must be integers"
    # directional finite difference agrees (no order flips at h = 1e-7)
    d = torch.randn(x.shape, generator=torch.Generator().manual_seed(3),
                    dtype=torch.float64)
    w = torch.randn(y.shape, generator=torch.Generator().manual_seed(4),
                    dtype=torch.float64)
    (gw,) = torch.autograd.grad((y * w).sum(), x)
    h = 1e-7
    with torch.no_grad():
        fd = ((yolo.median_differentiable(x + h * d, 3) * w).sum()
              - (yolo.median_differentiable(x, 3) * w).sum()) / h
    assert abs(fd.item() - (gw * d).sum().item()) < 1e-5, (fd.item(),
                                                          (gw * d).sum().item())


check("median surrogate gradient: non-zero, one input per output, matches FD",
      t_median_gradient)


def t_gaussian():
    x = (0.2 + 0.6 * torch.rand(1, 3, 33, 30,
                                generator=torch.Generator().manual_seed(5)))
    diff = (yolo.gaussian_differentiable(x, 1.0)
            - pc.gaussian_gpu(x, sigma=1.0)).abs().max().item()
    assert diff < 1e-6, diff
    xr = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(yolo.gaussian_differentiable(xr, 1.0).sum(), xr)
    assert g.abs().sum() > 0
    yolo.assert_gaussian_matches(CPU, 1.0)


check("gaussian_differentiable == pc.gaussian_gpu, gradient non-zero",
      t_gaussian)


def t_deployed_value_matches_build_branches():
    """The surrogate's forward is bit-identical to what is actually scored."""
    args = make_args()
    pil = rand_pil(45, 37, 6)
    names = list(yolo.SURROGATE_BRANCHES)
    deployed = core.build_branches(pil, CPU, args, names)   # phase3_common
    x = pc.pil_to_tensor(pil, CPU)
    for b in names:
        mine = yolo.deployed_value(x, b, args)
        ref = pc.pil_to_tensor(deployed[b], CPU)
        diff = (mine - ref).abs().max().item()
        assert diff == 0.0, f"{b}: deployed_value differs by {diff * 255:.2f}/255"
    # and with the TVM-free producer path (no tvm/blur_tvm requested)
    deployed2 = core.build_branches(pil, CPU, args, ["jpeg", "median",
                                                     "gaussian"])
    for b in ("jpeg", "median", "gaussian"):
        assert torch.equal(yolo.deployed_value(x, b, args),
                           pc.pil_to_tensor(deployed2[b], CPU)), b


check("deployed_value == p4_core.build_branches for all 5 surrogate branches",
      t_deployed_value_matches_build_branches)


def t_bpda_backward():
    pil = rand_pil(30, 26, 7)
    x0 = pc.pil_to_tensor(pil, CPU)
    w = torch.randn(x0.shape, generator=torch.Generator().manual_seed(8))

    def grad_of(fn, x):
        xr = x.clone().requires_grad_(True)
        (g,) = torch.autograd.grad((fn(xr) * w).sum(), xr)
        return g

    for median_mode in ("true", "identity"):
        args = make_args("--median-surrogate", median_mode)
        for b in yolo.SURROGATE_BRANCHES:
            out = yolo.bpda_branch(x0.clone().requires_grad_(True), b, args)
            assert torch.equal(out.detach(), yolo.deployed_value(x0, b, args)), b
            g = grad_of(lambda t: yolo.bpda_branch(t, b, args), x0)
            if b in ("gaussian", "blur_tvm"):
                ref = grad_of(lambda t: yolo.gaussian_differentiable(t, 1.0), x0)
            elif b == "median" and median_mode == "true":
                ref = grad_of(lambda t: yolo.median_differentiable(t, 3), x0)
            else:
                ref = w                                  # identity Jacobian
            assert torch.allclose(g, ref, atol=1e-6), (b, median_mode)
            assert yolo.is_identity_surrogate(b, args) == (ref is w), b


check("bpda_branch: deployed forward; identity / true-median / Gaussian backward",
      t_bpda_backward)

print("\n[2] objective and union-aware EOT loss on fake [1,84,8400] heads")


def _fake_preds(k, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.cat([torch.randn(1, 4, 8400, generator=g) * 50,
                       torch.rand(1, 80, 8400, generator=g)], dim=1)
            for _ in range(k)]


def t_anchor_scores():
    (p,) = _fake_preds(1, 9)
    s = yolo.anchor_scores_from_pred(p)
    assert s.shape == (1, 8400)
    assert torch.equal(s, p[:, 4:, :].max(dim=1).values)
    q = p.clone()
    q[:, :4, :] = 1e6                       # box rows must not matter
    assert torch.equal(yolo.anchor_scores_from_pred(q), s)


check("anchor score = max over the 80 class rows; box rows ignored",
      t_anchor_scores)


def t_union_max_loss():
    preds = [p.requires_grad_(True) for p in _fake_preds(3, 10)]
    S = torch.stack([yolo.anchor_scores_from_pred(p) for p in preds])  # [3,1,A]
    P = torch.stack([p.detach()[0, 4:, :] for p in preds])           # [3,80,A]
    v_max = yolo.union_objective(S, "max")
    assert abs(v_max.item() - P.amax(dim=(0, 1)).sum().item()) < 1e-3
    v_mean = yolo.union_objective(S, "mean")
    ref_mean = sum(P[b].max(dim=0).values.sum().item() for b in range(3)) / 3
    assert abs(v_mean.item() - ref_mean) < 1e-3
    assert v_max.item() > v_mean.item(), "per-anchor max >= mean of branches"
    grads = torch.autograd.grad(v_max, preds)
    G = torch.stack([g[0] for g in grads])                            # [3,84,A]
    assert G[:, :4, :].abs().sum() == 0, "no gradient into the box rows"
    assert torch.equal(G.sum(dim=(0, 1)), torch.ones(8400)), \
        "exactly one (branch, class) receives each anchor's gradient"
    W = yolo.union_weights(S.detach())
    assert torch.equal(G.sum(dim=1), W[:, 0, :]), \
        "union_weights must equal d(max loss)/d(scores)"
    winners = set(W.argmax(dim=0).flatten().tolist())
    assert winners == {0, 1, 2}, f"degenerate test: winners {winners}"
    try:
        yolo.union_objective(S, "sum")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown --eot-loss must raise")


check("max-over-branches loss: value, one-hot gradient == union_weights",
      t_union_max_loss)


def _eot_reference(model, frame, branches, args, adv):
    """Single backward through all k stacked graphs (the textbook EOT)."""
    values = [yolo.deployed_value(adv, b, args) for b in branches]
    x = adv.clone().requires_grad_(True)
    S = torch.stack([yolo.yolo_anchor_scores(
        model, frame.to_model(yolo.bpda_branch(x, b, args, v)))
        for b, v in zip(branches, values)])
    L = yolo.union_objective(S, args.eot_loss)
    (g,) = torch.autograd.grad(L, x)
    return L.item(), g, S.detach()


def t_eot_gradient_equals_single_backward():
    pil = rand_pil(40, 30, 11)
    tier1 = pc.TIER1_ENSEMBLES
    cases = 0
    combos = [(t, lo, m, e)
              for t in ("ens_jpeg_median_gaussian", "ens_4way")
              for lo in ("max", "mean") for m in ("true", "identity")
              for e in ("on", "off")]
    for target, loss, med, mode in combos:
        args = make_args("--target", target, "--eot-loss", loss,
                         "--median-surrogate", med, "--eot-batch", mode)
        _, branches, _ = yolo.resolve_target(target)
        assert branches == tier1[target]
        frame = yolo.attack_frame(pil, CPU, args, MODEL)
        adv = (frame.x0 + yolo.pgd_init_noise(frame.x0, 0.03, 0, 1)
               ).clamp(0, 1)
        val, g = yolo.eot_objective(MODEL, frame, branches, args)(
            adv, need_grad=True)
        ref_val, ref_g, S = _eot_reference(MODEL, frame, branches, args, adv)
        case = (target, loss, med, mode)
        assert abs(val - ref_val) < 1e-4 * max(1.0, abs(ref_val)), \
            (case, val, ref_val)
        assert torch.allclose(g, ref_g, atol=1e-6, rtol=1e-4), \
            (case, (g - ref_g).abs().max().item())
        if loss == "max":
            won = set(yolo.union_weights(S).argmax(0).flatten().tolist())
            assert len(won) >= 2, f"degenerate: one branch won all {won}"
        nograd, none = yolo.eot_objective(MODEL, frame, branches,
                                          args)(adv, need_grad=False)
        assert none is None and abs(nograd - ref_val) < 1e-4 * abs(ref_val)
        cases += 1
    assert cases == 16


check("eot_objective (batched and branch by branch) == one backward over "
      "stacked graphs (EnsJMG/Ens4W x max/mean x median true/identity x "
      "--eot-batch on/off)",
      t_eot_gradient_equals_single_backward)


def _old_deployed_value(x, branch, args):
    """deployed_value as it was before the shared round trip (verbatim)."""
    q = yolo._quantize_uint8
    with torch.no_grad():
        if branch == "jpeg":
            return pc.pil_to_tensor(
                pc.jpeg_cpu(pc.tensor_to_pil(x), args.jpeg_quality), x.device)
        if branch == "median":
            return q(pc.median_gpu(q(x), args.median_kernel))
        if branch == "gaussian":
            return q(pc.gaussian_gpu(q(x), sigma=args.gaussian_sigma))
        if branch == "tvm":
            return q(pc.tvm_gpu(q(x), weight=args.tvm_weight,
                                n_iter=args.tvm_iters))
        if branch == "blur_tvm":
            return q(pc.tvm_gpu(pc.gaussian_gpu(q(x), sigma=args.gaussian_sigma),
                                weight=args.tvm_weight, n_iter=args.tvm_iters))
    raise ValueError(branch)


def t_deployed_values_shared_round_trip():
    """Sharing the input's uint8 round trip changes no bit of any branch."""
    names = list(yolo.SURROGATE_BRANCHES)
    checked = 0
    for space in ("letterbox", "native"):
        args = make_args("--attack-space", space)
        for seed in (21, 22):
            pil = rand_pil(45, 37, seed)
            frame = yolo.attack_frame(pil, CPU, args, MODEL)
            # an off-grid (non-uint8) input, as every PGD iterate is
            adv = (frame.x0 + yolo.pgd_init_noise(frame.x0, 0.03, seed, 5)
                   ).clamp(0, 1)
            for x in (frame.x0, adv):
                shared = yolo.deployed_values(x, names, args)
                for b, v in zip(names, shared):
                    assert torch.equal(v, _old_deployed_value(x, b, args)), \
                        (space, seed, b)
                    assert torch.equal(v, yolo.deployed_value(x, b, args)), b
                    checked += 1
                # any branch order / subset gives the same tensors
                sub = yolo.deployed_values(x, ["gaussian", "jpeg"], args)
                assert torch.equal(sub[0], shared[names.index("gaussian")])
                assert torch.equal(sub[1], shared[names.index("jpeg")])
    assert checked == 2 * 2 * 2 * len(names)
    try:
        yolo.deployed_values(frame.x0, ["svd"], args)
    except ValueError:
        return
    raise AssertionError("a branch without a surrogate must raise")


check("deployed_values (one shared uint8 round trip) == the pre-change "
      "deployed_value, bit for bit, letterbox + native, clean + off-grid",
      t_deployed_values_shared_round_trip)

print("\n[3] seeded init, restarts, quantization, PGD")


def t_seeded_init_shared():
    x0 = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(12))
    a = yolo.pgd_init_noise(x0, 0.03, 42, 139, 0)
    assert torch.equal(a, yolo.pgd_init_noise(x0, 0.03, 42, 139, 0))
    assert a.abs().max() <= 0.03 and a.abs().max() > 0.025
    for other in (yolo.pgd_init_noise(x0, 0.03, 42, 285, 0),
                  yolo.pgd_init_noise(x0, 0.03, 42, 139, 1),
                  yolo.pgd_init_noise(x0, 0.03, 7, 139, 0)):
        assert not torch.equal(a, other), "image id / restart / seed ignored"
    torch.manual_seed(999)                    # global RNG must not matter
    assert torch.equal(a, yolo.pgd_init_noise(x0, 0.03, 42, 139, 0))

    # the two attacks start from the SAME draw: with 0 iterations their
    # outputs are exactly that draw, quantized
    pil = rand_pil(40, 30, 13)
    args = make_args("--pgd-iters", "1")
    args.pgd_iters = 0
    _, branches, _ = yolo.resolve_target(args.target)
    pil_o, obl = yolo.pgd_oblivious(pil, MODEL, CPU, args, 139)
    pil_a, adp = yolo.pgd_adaptive_bpda_eot(pil, MODEL, CPU, args, branches, 139)
    assert torch.equal(obl["adv_q"], adp["adv_q"]), "attacks start apart"
    assert np.array_equal(np.array(pil_o), np.array(pil_a))
    frame = yolo.attack_frame(pil, CPU, args, MODEL)
    start = (frame.x0 + yolo.pgd_init_noise(frame.x0, args.eps, args.seed, 139)
             ).clamp(0, 1)
    assert torch.equal(obl["adv_q"], yolo.quantize_adv(start, "truncate")[1])
    _, obl2 = yolo.pgd_oblivious(pil, MODEL, CPU, args, 285)
    assert not torch.equal(obl["adv_q"], obl2["adv_q"]), "image id ignored"


check("seeded PGD init: per (seed, image id, restart); both attacks share it",
      t_seeded_init_shared)


def t_quantize_adv():
    adv = torch.tensor([0.0, 0.5 / 255, 1.6 / 255, 254.7 / 255, 1.0]
                       ).view(1, 1, 1, 5).expand(1, 3, 1, 5).contiguous()
    u8t, qt = yolo.quantize_adv(adv, "truncate")
    u8r, qr = yolo.quantize_adv(adv, "round")
    assert u8t[0, :, 0].tolist() == [0, 0, 1, 254, 255], u8t[0, :, 0]
    assert u8r[0, :, 0].tolist() == [0, 0, 2, 255, 255], u8r[0, :, 0]
    assert u8t.dtype == np.uint8 and u8t.shape == (1, 5, 3)
    assert torch.allclose(qr * 255, torch.tensor(
        u8r.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0))
    assert (qt <= adv + 1e-7).all(), "truncation can only floor"
    # eps = 0.03: a +/-7.65/255 step truncates to -8 / rounds to +/-8;
    # with x0 the result is clipped back to the whole budget, +/-7
    u0 = np.array([0, 100, 100, 200, 255], np.float32)
    x0 = torch.from_numpy(u0 / 255).view(1, 1, 1, 5).expand(1, 3, 1, 5
                                                            ).contiguous()
    step = torch.tensor([7.65, -7.65, 7.65, -3.2, -7.65]
                        ).view(1, 1, 1, 5) / 255
    adv2 = (x0 + step).clamp(0, 1)
    assert yolo.quantize_adv(adv2, "truncate")[0][0, :, 0].tolist() == \
        [7, 92, 107, 196, 247]
    assert yolo.quantize_adv(adv2, "round")[0][0, :, 0].tolist() == \
        [8, 92, 108, 197, 247]
    for mode in ("truncate", "round"):
        u8c, qc = yolo.quantize_adv(adv2, mode, x0, 0.03)
        d = u8c[0, :, 0].astype(int) - u0.astype(int)
        assert np.abs(d).max() == 7, (mode, d)
        assert torch.allclose(qc * 255, torch.tensor(
            u8c.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0))


check("quantize_adv: truncate floors, round rounds, tensor == uint8/255, "
      "x0 clip keeps eps",
      t_quantize_adv)


def t_pgd_and_restarts():
    pil = rand_pil(40, 30, 14)
    args = make_args("--pgd-iters", "8", "--eps", "0.06")
    _, branches, _ = yolo.resolve_target(args.target)
    frame = yolo.attack_frame(pil, CPU, args, MODEL)
    clean_bare, _ = yolo.oblivious_objective(MODEL, frame)(frame.x0, False)
    _, obl = yolo.pgd_oblivious(pil, MODEL, CPU, args, 139)
    assert obl["objective"] < clean_bare, "oblivious PGD did not lower J"
    # letterbox keeps Phase 3's conversion: truncation can floor a -15.3/255
    # step to -16/255
    assert ((obl["adv_q"] - frame.x0) * 255).abs().max() <= \
        np.ceil(args.eps * 255) + 1e-3
    _, adp = yolo.pgd_adaptive_bpda_eot(pil, MODEL, CPU, args, branches, 139)
    tj = yolo.target_objectives(pil, MODEL, CPU, args, branches,
                                {"clean": None, "oblivious": obl["adv_q"]})
    assert adp["objective"] < tj["clean"], "adaptive PGD did not lower its J"
    # restarts: R=3 keeps the argmin; restart 0 is the same draw as with R=1
    args3 = make_args("--pgd-iters", "8", "--eps", "0.06", "--restarts", "3")
    _, adp3 = yolo.pgd_adaptive_bpda_eot(pil, MODEL, CPU, args3, branches, 139)
    objs = adp3["restart_objectives"]
    assert len(objs) == 3 and len(set(objs)) == 3, objs
    assert adp3["objective"] == min(objs) and adp3["restart"] == objs.index(
        min(objs))
    assert abs(objs[0] - adp["objective"]) < 1e-6 * abs(objs[0]), \
        "restart 0 must not depend on --restarts"
    assert adp3["objective"] <= adp["objective"]
    # --pgd-select best: never worse than last on the quantized objective,
    # for both attacks, and the default stays `last`
    assert args.pgd_select == "last"
    argsb = make_args("--pgd-iters", "8", "--eps", "0.06", "--pgd-select",
                      "best")
    _, oblb = yolo.pgd_oblivious(pil, MODEL, CPU, argsb, 139)
    _, adpb = yolo.pgd_adaptive_bpda_eot(pil, MODEL, CPU, argsb, branches, 139)
    assert oblb["objective"] <= obl["objective"] + 1e-9
    assert adpb["objective"] <= adp["objective"] + 1e-9
    frame_b = yolo.attack_frame(pil, CPU, argsb, MODEL)
    re_j, _ = yolo.eot_objective(MODEL, frame_b, branches, argsb)(
        adpb["adv_q"], False)
    assert abs(re_j - adpb["objective"]) < 1e-6 * abs(re_j), \
        "the returned tensor must be the one the objective was measured on"


check("PGD lowers J (fake model); --restarts keeps the lowest-J restart",
      t_pgd_and_restarts)


def t_check_surrogates_guard():
    for target in yolo.ADAPTIVE_TARGETS:
        args = make_args("--target", target)
        _, branches, _ = yolo.resolve_target(target)
        yolo.check_surrogates(CPU, args, branches)


check("check_surrogates passes for every --target on CPU",
      t_check_surrogates_guard)

print("\n[4] --attack-space native: differentiable ultralytics letterbox")


def t_native_letterbox_vs_ultralytics():
    try:
        from ultralytics.data.augment import LetterBox
    except Exception as exc:                  # noqa: BLE001
        raise Skip(f"ultralytics not importable: {exc!r}")
    exact, resized = 0, 0
    for (w, h) in ((640, 470), (427, 640), (640, 640), (500, 375), (800, 533),
                   (333, 500)):
        img = np.asarray(rand_pil(w, h, w + h))
        ref = LetterBox((640, 640), auto=True, stride=32)(image=img)
        t = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1
                                                                   ).unsqueeze(0)
        mine = yolo.letterbox_differentiable(t, 640, stride=32)
        mine = (mine[0].permute(1, 2, 0).numpy() * 255)
        assert mine.shape == ref.shape, ((w, h), mine.shape, ref.shape)
        assert ref.shape[0] % 32 == 0 and ref.shape[1] % 32 == 0
        diff = np.abs(mine - ref.astype(np.float32))
        if max(w, h) == 640:                  # no resize: exact
            assert diff.max() < 1e-3, ((w, h), diff.max())
            exact += 1
        else:                                 # torch bilinear vs cv2 uint8
            # measured 24 Oct: max 0.75-0.79/255, mean ~0.26/255 (cv2 rounds
            # its output to uint8; torch stays float)
            assert diff.max() <= 1.0 and diff.mean() < 0.5, \
                ((w, h), diff.max(), diff.mean())
            resized += 1
    assert (exact, resized) == (3, 3)


check("letterbox_differentiable == ultralytics LetterBox(auto, stride 32) "
      "(exact w/o resize; < 1/255 with resize)",
      t_native_letterbox_vs_ultralytics)


def t_native_frame_and_attack():
    pil = rand_pil(100, 30, 15)
    args = yolo.parse_args(["--experiment", "adaptive_bpda", "--yolo-imgsz",
                            "64", "--attack-space", "native", "--pgd-iters",
                            "3"])
    frame = yolo.attack_frame(pil, CPU, args, MODEL)
    assert frame.space == "native" and tuple(frame.x0.shape) == (1, 3, 30, 100)
    xm = frame.to_model(frame.x0)
    assert tuple(xm.shape) == (1, 3, 32, 64), xm.shape   # rect, stride 32
    assert torch.allclose(xm[..., :6, :], torch.full_like(xm[..., :6, :],
                                                          114 / 255))
    xr = frame.x0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(frame.to_model(xr).sum(), xr)
    assert g.abs().sum() > 0, "no gradient through the native letterbox"
    _, branches, _ = yolo.resolve_target(args.target)
    adv_pil, info = yolo.pgd_adaptive_bpda_eot(pil, MODEL, CPU, args,
                                               branches, 139)
    assert adv_pil.size == pil.size, "native output must keep native size"
    d = np.abs(np.asarray(adv_pil, np.int16) - np.asarray(pil, np.int16))
    assert d.max() <= yolo.uint8_budget(args.eps), d.max()  # within eps
    assert np.array_equal(np.asarray(adv_pil),
                          yolo.quantize_adv(info["adv_q"], "truncate")[0])


check("native frame: rect letterbox into the model, attack output at native "
      "size within eps (7/255 at eps 0.03)", t_native_frame_and_attack)


print("\n[5] --eot-batch: one batched forward + backward == branch by branch")


def _k_branches():
    """k = 1..4 branch lists; 3 and 4 are EnsJMG / Ens4W from phase3_common."""
    return {1: ["median"], 2: ["jpeg", "gaussian"],
            3: yolo.resolve_target("ens_jpeg_median_gaussian")[1],
            4: yolo.resolve_target("ens_4way")[1]}


class _CountingDetector(_FakeDetector):
    """The fake detector (same weights) recording each forward's batch size;
    optionally raising CUDA OOM for batches larger than `oom_above`."""

    def __init__(self, oom_above=None):
        super().__init__()
        self.batches, self.oom_above = [], oom_above

    def forward(self, x):
        self.batches.append(int(x.shape[0]))
        if self.oom_above is not None and x.shape[0] > self.oom_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return super().forward(x)


class _CountingYolo:
    def __init__(self, oom_above=None):
        self.model = _CountingDetector(oom_above).eval()


def _scores_both_ways(frame, branches, args, adv):
    """(S from k separate forwards, S from one batched forward), no grad."""
    vals = yolo.deployed_values(adv, branches, args)
    with torch.no_grad():
        s_seq = torch.stack([yolo.yolo_anchor_scores(MODEL, frame.to_model(v))
                             for v in vals])
        s_bat = yolo.yolo_anchor_scores(
            MODEL, torch.cat([frame.to_model(v) for v in vals])
        ).view(len(branches), 1, -1)
    return s_seq, s_bat


def _near_tie_flip(s_seq, s_bat, tol=1e-6):
    """True if the winner masks differ only at anchors whose two best branch
    scores are within `tol`: a float-rounding tie, not a bug."""
    w_seq = yolo.union_weights(s_seq).argmax(0)
    w_bat = yolo.union_weights(s_bat).argmax(0)
    flipped = (w_seq != w_bat)
    if not flipped.any():
        return False
    top2 = s_seq.topk(2, dim=0).values                  # [2, B, A]
    gap = (top2[0] - top2[1])[flipped]
    return bool((gap <= tol).all())


def t_eot_batch_matches_sequential():
    """Same J and gradient, k = 1..4, mean and max, letterbox and native."""
    cases, ties = 0, 0
    for k, branches in _k_branches().items():
        for loss in ("mean", "max"):
            for space in ("letterbox", "native"):
                on = make_args("--eot-loss", loss, "--attack-space", space,
                               "--eot-batch", "on")
                off = make_args("--eot-loss", loss, "--attack-space", space,
                                "--eot-batch", "off")
                pil = rand_pil(40, 30, 30 + k)
                frame = yolo.attack_frame(pil, CPU, on, MODEL)
                adv = (frame.x0 + yolo.pgd_init_noise(frame.x0, 0.03, k, 3)
                       ).clamp(0, 1)
                case = (k, loss, space)
                v_on, g_on = yolo.eot_objective(MODEL, frame, branches, on)(
                    adv, True)
                v_off, g_off = yolo.eot_objective(MODEL, frame, branches,
                                                  off)(adv, True)
                n_on, _ = yolo.eot_objective(MODEL, frame, branches, on)(
                    adv, False)
                n_off, _ = yolo.eot_objective(MODEL, frame, branches, off)(
                    adv, False)
                for a, b in ((v_on, v_off), (n_on, n_off), (v_on, n_off)):
                    assert abs(a - b) <= 1e-5 * abs(b), (case, a, b)
                if k == 1:            # nothing to batch: the same code path
                    assert torch.equal(g_on, g_off) and v_on == v_off, case
                if loss == "max" and k > 1 and _near_tie_flip(
                        *_scores_both_ways(frame, branches, on, adv)):
                    ties += 1         # a different, equally exact subgradient
                    continue
                assert torch.allclose(g_on, g_off, rtol=1e-4, atol=1e-6), \
                    (case, (g_on - g_off).abs().max().item())
                big = g_off.abs() > 1e-6
                assert big.sum() > 100, (case, "degenerate gradient")
                assert torch.equal(g_on.sign()[big], g_off.sign()[big]), case
                cases += 1
    assert cases + ties == 16 and ties <= 2, (cases, ties)


check("--eot-batch on == off: J within 1e-5, gradients allclose, same signs "
      "where |g| > 1e-6 (k = 1..4 x mean/max x letterbox/native)",
      t_eot_batch_matches_sequential)


def t_eot_batch_forward_count():
    """on = ONE batched forward per call, and no separate no-grad pass."""
    branches = _k_branches()[3]
    pil = rand_pil(40, 30, 40)
    for loss in ("max", "mean"):
        for mode in ("on", "off"):
            args = make_args("--eot-loss", loss, "--eot-batch", mode)
            model = _CountingYolo()
            frame = yolo.attack_frame(pil, CPU, args, model)
            adv = (frame.x0 + yolo.pgd_init_noise(frame.x0, 0.03, 0, 9)
                   ).clamp(0, 1)
            obj = yolo.eot_objective(model, frame, branches, args)
            obj(adv, True)
            grad_calls = list(model.model.batches)
            model.model.batches.clear()
            obj(adv, False)
            value_calls = list(model.model.batches)
            if mode == "on":
                assert grad_calls == [3] and value_calls == [3], \
                    (loss, grad_calls, value_calls)
            else:
                assert value_calls == [1, 1, 1], value_calls
                assert set(grad_calls) == {1}
                # max: k no-grad winner passes + one graph per winning branch
                assert len(grad_calls) == (3 + 3 if loss == "max" else 3), \
                    (loss, grad_calls)


check("--eot-batch on: one [k,3,H,W] forward per call (max: no extra no-grad "
      "pass); off: branch by branch", t_eot_batch_forward_count)


def t_eot_batch_pgd_same_image():
    """A full PGD run gives the same quantized image on vs off, except where a
    float-rounding tie flips the max loss's winning branch (explained)."""
    same, explained = 0, 0
    for k, branches in _k_branches().items():
        for loss in ("mean", "max"):
            for space in ("letterbox", "native"):
                common = ["--eot-loss", loss, "--attack-space", space,
                          "--pgd-iters", "10"]
                on = make_args(*common, "--eot-batch", "on")
                off = make_args(*common, "--eot-batch", "off")
                pil = rand_pil(40, 30, 11)
                _, i_on = yolo.pgd_adaptive_bpda_eot(pil, MODEL, CPU, on,
                                                     branches, 139)
                _, i_off = yolo.pgd_adaptive_bpda_eot(pil, MODEL, CPU, off,
                                                      branches, 139)
                case = (k, loss, space)
                assert abs(i_on["objective"] - i_off["objective"]) <= \
                    1e-5 * abs(i_off["objective"]), case
                if torch.equal(i_on["adv_q"], i_off["adv_q"]):
                    same += 1
                    continue
                # Replay both trajectories in lockstep (run_pgd's step) to the
                # first step whose gradient signs differ on the SAME input,
                # and require the reason to be a winner flip at a near tie
                # (both are exact subgradients of max at a tie) or sum-order
                # rounding of components that are ~0 (|g| <= 1e-6).
                assert k > 1, case
                frame = yolo.attack_frame(pil, CPU, on, MODEL)
                x0 = frame.x0
                adv = (x0 + yolo.pgd_init_noise(x0, on.eps, on.seed, 139)
                       ).clamp(0, 1)
                o_on = yolo.eot_objective(MODEL, frame, branches, on)
                o_off = yolo.eot_objective(MODEL, frame, branches, off)
                for _ in range(on.pgd_iters):
                    _, g1 = o_on(adv, True)
                    _, g2 = o_off(adv, True)
                    diff = g1.sign() != g2.sign()
                    if diff.any():
                        tiny = bool((g2.abs()[diff] <= 1e-6).all())
                        tie = loss == "max" and _near_tie_flip(
                            *_scores_both_ways(frame, branches, on, adv))
                        assert tiny or tie, (case, "unexplained divergence")
                        explained += 1
                        break
                    with torch.no_grad():
                        adv = adv - on.alpha * g1.sign()
                        adv = torch.clamp(x0 + torch.clamp(
                            adv - x0, -on.eps, on.eps), 0.0, 1.0)
                else:
                    raise AssertionError((case, "images differ, steps agree"))
    print(f"        (identical quantized image in {same}/16 runs; "
          f"{explained} explained near-tie / ~0-gradient divergence(s))")
    assert same + explained == 16 and same >= 12, (same, explained)


check("--eot-batch on vs off: full PGD gives the same quantized image (any "
      "difference traced to a near-tie winner flip)", t_eot_batch_pgd_same_image)


def t_eot_batch_oom_fallback():
    """A CUDA OOM in the batched path switches the run to branch by branch."""
    branches = _k_branches()[3]
    pil = rand_pil(40, 30, 41)
    args = make_args("--eot-batch", "on")
    ref_args = make_args("--eot-batch", "off")
    model = _CountingYolo(oom_above=1)              # any batch > 1 "OOMs"
    frame = yolo.attack_frame(pil, CPU, args, model)
    adv = (frame.x0 + yolo.pgd_init_noise(frame.x0, 0.03, 0, 4)).clamp(0, 1)
    v, g = yolo.eot_objective(model, frame, branches, args)(adv, True)
    v_ref, g_ref = yolo.eot_objective(MODEL, frame, branches, ref_args)(
        adv, True)
    assert v == v_ref and torch.equal(g, g_ref), "fallback is the off path"
    assert args.eot_batch_oom_fallback is True and args.eot_batch_mode == "on"
    assert yolo.ensure_eot_batch(args, CPU) == "off", "stays off for the run"
    model.model.batches.clear()
    yolo.eot_objective(model, frame, branches, args)(adv, False)
    assert model.model.batches == [1, 1, 1], model.model.batches
    rec = yolo.execution_record(args, branches)
    assert rec["eot_batch"] == "on" and rec["eot_batch_oom_fallback"] is True
    # a non-OOM error is NOT swallowed
    bad = make_args("--eot-batch", "on")

    class _Broken(_CountingDetector):
        def forward(self, x):
            raise RuntimeError("shape mismatch in conv")

    broken = types.SimpleNamespace(model=_Broken().eval())
    try:
        yolo.eot_objective(broken, frame, branches, bad)(adv, True)
    except RuntimeError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("a non-OOM RuntimeError must propagate")


check("--eot-batch: CUDA OOM falls back to the sequential path for the rest "
      "of the run (recorded); other errors propagate", t_eot_batch_oom_fallback)


def t_eot_batch_resolution():
    assert yolo.resolve_eot_batch("auto", CPU)[0] == "on"
    gib = 2 ** 30
    cuda = torch.device("cuda", 0)            # only a label: no CUDA call
    assert yolo.resolve_eot_batch("auto", cuda, 48 * gib)[0] == "on"
    assert yolo.resolve_eot_batch("auto", cuda, 16 * gib)[0] == "on"
    assert yolo.resolve_eot_batch("auto", cuda, 6 * gib)[0] == "off"
    assert yolo.resolve_eot_batch("off", CPU)[0] == "off"
    assert yolo.resolve_eot_batch("on", cuda, 1)[0] == "on"
    a = make_args()
    assert a.eot_batch == "auto" and not hasattr(a, "eot_batch_mode")
    assert yolo.ensure_eot_batch(a, CPU) == "on"
    assert a.eot_batch_reason.startswith("auto")


check("--eot-batch auto: on for CPU or >= 16 GiB, else off; resolved once",
      t_eot_batch_resolution)


print("\n" + "=" * 60)
print(f"Phase-4 attack unit tests: {len(PASS)} passed, {len(FAIL)} failed, "
      f"{len(SKIP)} skipped")
if FAIL:
    print("FAILED:", FAIL)
print("=" * 60)
sys.exit(1 if FAIL else 0)
