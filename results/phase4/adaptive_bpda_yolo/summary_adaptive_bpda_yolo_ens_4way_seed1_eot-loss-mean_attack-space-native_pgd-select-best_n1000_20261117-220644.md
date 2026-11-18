# Adaptive white-box pilot: BPDA + EOT vs Ens4W (yolov8x-worldv2.pt)

- Run `20261117-220644` · model `yolo` · experiment `adaptive_bpda`
- Images: **n = 1000** (`--select head`, seed 1, id digest `620b79f17bd75ab3`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Target **Ens4W** (`ens_4way`) = jpeg + median + tvm + gaussian, merged by class-aware NMS @ IoU 0.5
- PGD eps=0.03, 50 iters, alpha=0.00200, 1 restart(s), raw [0,1] space; seeded init per (seed 1, image id, restart), shared by both attacks
- Attack space `native` · output quantization `truncate` · median surrogate `true` · iterate selection `best`
- EOT loss `mean`: mean of the 4 branch objectives (per-step gradient = mean of the branch gradients)
- Execution: EOT batch `on` (auto: 31.5 GiB >= 16 GiB) (an execution choice: same objective, equal up to float rounding)
- BPDA surrogates (backward only; every forward is the deployed value): jpeg: identity (BPDA straight-through); median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected); tvm: identity (BPDA straight-through); gaussian: true gradient
- All headline numbers come from the SAME image subset.

## Headline (COCO mAP@[.5:.95])

| Condition | mAP |
|---|---|
| clean, undefended | 0.4764 |
| clean, Ens4W | 0.4684 |
| oblivious attack, undefended | 0.0183 |
| oblivious attack, Ens4W | 0.4312 |
| adaptive attack, undefended | 0.0518 |
| adaptive attack, Ens4W | 0.0687 |

- Defense recovery vs an oblivious attacker: **+0.4129**
- Defense recovery vs an adaptive attacker: **+0.0169**
- Defended mAP lost to adaptivity: **-0.3624**

## Surrogate objective (the attacker's own view)

J = sum over anchors of the max-class confidence, measured THROUGH Ens4W with the attack's own surrogate (deployed forward values, native frame, eot-loss mean); lower = more suppression. A diagnostic of the attacker's optimisation, not a detection metric.

| Input | mean J through the target |
|---|---|
| clean | 38.9813 |
| oblivious adversarial | 24.4580 |
| adaptive adversarial | 1.6280 |

- Images where the adaptive J is below the oblivious J: **1.000** of 1000 (should be close to 1; if not, the adaptive attack is not optimising its objective)

## All conditions

| Input | Defense | Tag | mAP | AP50 |
|---|---|---|---|---|
| clean | undefended | `clean` | 0.4764 | 0.6399 |
| clean | Ens4W | `clean+ens_4way` | 0.4684 | 0.6349 |
| clean | Gaussian | `clean+gaussian` | 0.4613 | 0.6270 |
| clean | JPEG | `clean+jpeg` | 0.4525 | 0.6154 |
| clean | Median | `clean+median` | 0.4583 | 0.6210 |
| clean | TVM | `clean+tvm` | 0.4472 | 0.6089 |
| adaptive attack | undefended | `pgd_adaptive` | 0.0518 | 0.0873 |
| adaptive attack | Ens4W | `pgd_adaptive+ens_4way` | 0.0687 | 0.1106 |
| adaptive attack | Gaussian | `pgd_adaptive+gaussian` | 0.0594 | 0.0961 |
| adaptive attack | JPEG | `pgd_adaptive+jpeg` | 0.0568 | 0.0931 |
| adaptive attack | Median | `pgd_adaptive+median` | 0.0288 | 0.0512 |
| adaptive attack | TVM | `pgd_adaptive+tvm` | 0.0658 | 0.1060 |
| oblivious attack | undefended | `pgd_oblivious` | 0.0183 | 0.0319 |
| oblivious attack | Ens4W | `pgd_oblivious+ens_4way` | 0.4312 | 0.5884 |
| oblivious attack | Gaussian | `pgd_oblivious+gaussian` | 0.4093 | 0.5628 |
| oblivious attack | JPEG | `pgd_oblivious+jpeg` | 0.3654 | 0.5076 |
| oblivious attack | Median | `pgd_oblivious+median` | 0.3691 | 0.5144 |
| oblivious attack | TVM | `pgd_oblivious+tvm` | 0.4221 | 0.5778 |

## Implementation uncertainty

- Native space: the perturbation lives at native resolution, each surrogate defense runs there as deployed, and the model input is a differentiable copy of ultralytics' predict letterbox (minimum-rectangle padding to the stride, grey 114, centred). It resizes with torch bilinear where ultralytics uses cv2 INTER_LINEAR on uint8, so the two agree exactly when no resize is needed (longest side == 640) and differ by interpolation rounding otherwise (under 1/255 per pixel against ultralytics' LetterBox in tests/test_attack_units.py). The letterbox-vs-native scale mismatch of the default mode is removed and nothing resamples the perturbation after the attack. The uint8 output is clipped to the clean image +/- 7/255 (eps*255 = 7.65 is not whole, and truncation or rounding alone would move a pixel by 8/255), so no pixel the defense receives moves more than eps.
- BPDA identity Jacobian (straight-through) for jpeg, tvm: the standard approximation, not the transform's true Jacobian; a learned differentiable proxy could give a stronger attacker.
- True gradients: median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected); gaussian: true gradient. The median gradient is sparse (one input per output pixel), so it carries less signal per step than a smooth filter's.
- Every surrogate forwards the deployed phase3_common transform's exact output on the attack-frame tensor (the native image), including the deployed pipeline's truncating uint8 round-trips (phase3_common.tensor_to_pil); those round-trips are straight-through in the backward pass. The attack output is converted to uint8 by truncation, the Phase-3 conversion: every perturbed pixel is floored, on average 0.5/255 below rounding.
- EOT loss 'mean' averages the 4 branch objectives; it ignores that the NMS-union merge keeps a box if ANY branch keeps it, so a box one branch still detects is not targeted specifically.
- EOT aggregates over deterministic defense branches, not over defense randomness (Ens4W has none); a randomized defense would need EOT samples per branch as well.
- PGD init: one seeded uniform(-eps, eps) draw per (seed, image id, restart), identical for the oblivious and the adaptive attack, so the two differ only in the gradient they follow. Phase 3 drew its init from the unseeded global RNG, so Phase-3 and Phase-4 attacked numbers also differ by init noise.
- PGD iterate selection 'best': per restart, the last iterate and the lowest-objective iterate seen are both quantized and the one with the lower objective after quantization is kept (Phase 3 kept the last iterate). Applied to the oblivious and the adaptive attack alike.

## Timing

- **13.74 s/image** on `cuda:0` — 228.9 min for 1000 of 1000 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 1.14 h · n=1000: 3.82 h · n=5000: 19.08 h
