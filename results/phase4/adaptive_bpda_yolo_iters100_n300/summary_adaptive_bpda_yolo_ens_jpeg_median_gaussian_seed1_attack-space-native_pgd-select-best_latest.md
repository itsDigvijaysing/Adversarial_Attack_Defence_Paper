# Adaptive white-box pilot: BPDA + EOT vs EnsJMG (yolov8x-worldv2.pt)

- Run `20261117-171529` · model `yolo` · experiment `adaptive_bpda`
- Images: **n = 300** (`--select head`, seed 1, id digest `94c8dc9df724d3df`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Target **EnsJMG** (`ens_jpeg_median_gaussian`) = jpeg + median + gaussian, merged by class-aware NMS @ IoU 0.5
- PGD eps=0.03, 100 iters, alpha=0.00200, 1 restart(s), raw [0,1] space; seeded init per (seed 1, image id, restart), shared by both attacks
- Attack space `native` · output quantization `truncate` · median surrogate `true` · iterate selection `best`
- EOT loss `max`: union-aware max: per anchor, max over the 3 branches of the max-class confidence, summed over anchors
- Execution: EOT batch `on` (auto: 31.5 GiB >= 16 GiB) (an execution choice: same objective, equal up to float rounding)
- BPDA surrogates (backward only; every forward is the deployed value): jpeg: identity (BPDA straight-through); median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected); gaussian: true gradient
- All headline numbers come from the SAME image subset.

## Headline (COCO mAP@[.5:.95])

| Condition | mAP |
|---|---|
| clean, undefended | 0.5032 |
| clean, EnsJMG | 0.4995 |
| oblivious attack, undefended | 0.0033 |
| oblivious attack, EnsJMG | 0.4428 |
| adaptive attack, undefended | 0.0604 |
| adaptive attack, EnsJMG | 0.0536 |

- Defense recovery vs an oblivious attacker: **+0.4395**
- Defense recovery vs an adaptive attacker: **-0.0068**
- Defended mAP lost to adaptivity: **-0.3892**

## Surrogate objective (the attacker's own view)

J = sum over anchors of the max-class confidence, measured THROUGH EnsJMG with the attack's own surrogate (deployed forward values, native frame, eot-loss max); lower = more suppression. A diagnostic of the attacker's optimisation, not a detection metric.

| Input | mean J through the target |
|---|---|
| clean | 51.2308 |
| oblivious adversarial | 33.6411 |
| adaptive adversarial | 0.7047 |

- Images where the adaptive J is below the oblivious J: **1.000** of 300 (should be close to 1; if not, the adaptive attack is not optimising its objective)

## All conditions

| Input | Defense | Tag | mAP | AP50 |
|---|---|---|---|---|
| clean | undefended | `clean` | 0.5032 | 0.6732 |
| clean | EnsJMG | `clean+ens_jpeg_median_gaussian` | 0.4995 | 0.6705 |
| clean | Gaussian | `clean+gaussian` | 0.4935 | 0.6641 |
| clean | JPEG | `clean+jpeg` | 0.4867 | 0.6554 |
| clean | Median | `clean+median` | 0.4891 | 0.6572 |
| adaptive attack | undefended | `pgd_adaptive` | 0.0604 | 0.0917 |
| adaptive attack | EnsJMG | `pgd_adaptive+ens_jpeg_median_gaussian` | 0.0536 | 0.0780 |
| adaptive attack | Gaussian | `pgd_adaptive+gaussian` | 0.0503 | 0.0709 |
| adaptive attack | JPEG | `pgd_adaptive+jpeg` | 0.0520 | 0.0767 |
| adaptive attack | Median | `pgd_adaptive+median` | 0.0323 | 0.0516 |
| oblivious attack | undefended | `pgd_oblivious` | 0.0033 | 0.0052 |
| oblivious attack | EnsJMG | `pgd_oblivious+ens_jpeg_median_gaussian` | 0.4428 | 0.6069 |
| oblivious attack | Gaussian | `pgd_oblivious+gaussian` | 0.4434 | 0.6114 |
| oblivious attack | JPEG | `pgd_oblivious+jpeg` | 0.3859 | 0.5354 |
| oblivious attack | Median | `pgd_oblivious+median` | 0.3985 | 0.5514 |

## Implementation uncertainty

- Native space: the perturbation lives at native resolution, each surrogate defense runs there as deployed, and the model input is a differentiable copy of ultralytics' predict letterbox (minimum-rectangle padding to the stride, grey 114, centred). It resizes with torch bilinear where ultralytics uses cv2 INTER_LINEAR on uint8, so the two agree exactly when no resize is needed (longest side == 640) and differ by interpolation rounding otherwise (under 1/255 per pixel against ultralytics' LetterBox in tests/test_attack_units.py). The letterbox-vs-native scale mismatch of the default mode is removed and nothing resamples the perturbation after the attack. The uint8 output is clipped to the clean image +/- 7/255 (eps*255 = 7.65 is not whole, and truncation or rounding alone would move a pixel by 8/255), so no pixel the defense receives moves more than eps.
- BPDA identity Jacobian (straight-through) for jpeg: the standard approximation, not the transform's true Jacobian; a learned differentiable proxy could give a stronger attacker.
- True gradients: median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected); gaussian: true gradient. The median gradient is sparse (one input per output pixel), so it carries less signal per step than a smooth filter's.
- Every surrogate forwards the deployed phase3_common transform's exact output on the attack-frame tensor (the native image), including the deployed pipeline's truncating uint8 round-trips (phase3_common.tensor_to_pil); those round-trips are straight-through in the backward pass. The attack output is converted to uint8 by truncation, the Phase-3 conversion: every perturbed pixel is floored, on average 0.5/255 below rounding.
- EOT loss 'max' mirrors the NMS-union merge (a box survives if ANY branch keeps it) by suppressing the most confident branch per anchor. It is still a proxy: the real merge matches post-threshold boxes across branches by IoU, not anchors by index. Its gradient is exact: per anchor, only the winning branch's gradient counts (computed in one batched pass or branch by branch, per --eot-batch, equal up to float rounding).
- EOT aggregates over deterministic defense branches, not over defense randomness (EnsJMG has none); a randomized defense would need EOT samples per branch as well.
- PGD init: one seeded uniform(-eps, eps) draw per (seed, image id, restart), identical for the oblivious and the adaptive attack, so the two differ only in the gradient they follow. Phase 3 drew its init from the unseeded global RNG, so Phase-3 and Phase-4 attacked numbers also differ by init noise.
- PGD iterate selection 'best': per restart, the last iterate and the lowest-objective iterate seen are both quantized and the one with the lower objective after quantization is kept (Phase 3 kept the last iterate). Applied to the oblivious and the adaptive attack alike.

## Timing

- **20.86 s/image** on `cuda:0` — 104.3 min for 300 of 300 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 1.74 h · n=1000: 5.80 h · n=5000: 28.98 h
