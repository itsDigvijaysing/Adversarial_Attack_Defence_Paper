# Adaptive white-box pilot: BPDA + EOT vs EnsJMG (yolov8x-worldv2.pt)

- Run `20261117-134742` · model `yolo` · experiment `adaptive_bpda`
- Images: **n = 1000** (`--select head`, seed 1, id digest `620b79f17bd75ab3`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Target **EnsJMG** (`ens_jpeg_median_gaussian`) = jpeg + median + gaussian, merged by class-aware NMS @ IoU 0.5
- PGD eps=0.03, 50 iters, alpha=0.00200, 1 restart(s), raw [0,1] space; seeded init per (seed 1, image id, restart), shared by both attacks
- Attack space `native` · output quantization `truncate` · median surrogate `true` · iterate selection `best`
- EOT loss `mean`: mean of the 3 branch objectives (per-step gradient = mean of the branch gradients)
- Execution: EOT batch `on` (auto: 31.5 GiB >= 16 GiB) (an execution choice: same objective, equal up to float rounding)
- BPDA surrogates (backward only; every forward is the deployed value): jpeg: identity (BPDA straight-through); median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected); gaussian: true gradient
- All headline numbers come from the SAME image subset.

## Headline (COCO mAP@[.5:.95])

| Condition | mAP |
|---|---|
| clean, undefended | 0.4764 |
| clean, EnsJMG | 0.4656 |
| oblivious attack, undefended | 0.0183 |
| oblivious attack, EnsJMG | 0.4106 |
| adaptive attack, undefended | 0.0817 |
| adaptive attack, EnsJMG | 0.0484 |

- Defense recovery vs an oblivious attacker: **+0.3923**
- Defense recovery vs an adaptive attacker: **-0.0334**
- Defended mAP lost to adaptivity: **-0.3623**

## Surrogate objective (the attacker's own view)

J = sum over anchors of the max-class confidence, measured THROUGH EnsJMG with the attack's own surrogate (deployed forward values, native frame, eot-loss mean); lower = more suppression. A diagnostic of the attacker's optimisation, not a detection metric.

| Input | mean J through the target |
|---|---|
| clean | 37.8748 |
| oblivious adversarial | 22.5319 |
| adaptive adversarial | 1.1159 |

- Images where the adaptive J is below the oblivious J: **1.000** of 1000 (should be close to 1; if not, the adaptive attack is not optimising its objective)

## All conditions

| Input | Defense | Tag | mAP | AP50 |
|---|---|---|---|---|
| clean | undefended | `clean` | 0.4764 | 0.6399 |
| clean | EnsJMG | `clean+ens_jpeg_median_gaussian` | 0.4656 | 0.6324 |
| clean | Gaussian | `clean+gaussian` | 0.4613 | 0.6270 |
| clean | JPEG | `clean+jpeg` | 0.4525 | 0.6154 |
| clean | Median | `clean+median` | 0.4583 | 0.6210 |
| adaptive attack | undefended | `pgd_adaptive` | 0.0817 | 0.1311 |
| adaptive attack | EnsJMG | `pgd_adaptive+ens_jpeg_median_gaussian` | 0.0484 | 0.0801 |
| adaptive attack | Gaussian | `pgd_adaptive+gaussian` | 0.0507 | 0.0821 |
| adaptive attack | JPEG | `pgd_adaptive+jpeg` | 0.0445 | 0.0754 |
| adaptive attack | Median | `pgd_adaptive+median` | 0.0253 | 0.0427 |
| oblivious attack | undefended | `pgd_oblivious` | 0.0183 | 0.0319 |
| oblivious attack | EnsJMG | `pgd_oblivious+ens_jpeg_median_gaussian` | 0.4106 | 0.5653 |
| oblivious attack | Gaussian | `pgd_oblivious+gaussian` | 0.4093 | 0.5628 |
| oblivious attack | JPEG | `pgd_oblivious+jpeg` | 0.3654 | 0.5076 |
| oblivious attack | Median | `pgd_oblivious+median` | 0.3691 | 0.5144 |

## Implementation uncertainty

- Native space: the perturbation lives at native resolution, each surrogate defense runs there as deployed, and the model input is a differentiable copy of ultralytics' predict letterbox (minimum-rectangle padding to the stride, grey 114, centred). It resizes with torch bilinear where ultralytics uses cv2 INTER_LINEAR on uint8, so the two agree exactly when no resize is needed (longest side == 640) and differ by interpolation rounding otherwise (under 1/255 per pixel against ultralytics' LetterBox in tests/test_attack_units.py). The letterbox-vs-native scale mismatch of the default mode is removed and nothing resamples the perturbation after the attack. The uint8 output is clipped to the clean image +/- 7/255 (eps*255 = 7.65 is not whole, and truncation or rounding alone would move a pixel by 8/255), so no pixel the defense receives moves more than eps.
- BPDA identity Jacobian (straight-through) for jpeg: the standard approximation, not the transform's true Jacobian; a learned differentiable proxy could give a stronger attacker.
- True gradients: median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected); gaussian: true gradient. The median gradient is sparse (one input per output pixel), so it carries less signal per step than a smooth filter's.
- Every surrogate forwards the deployed phase3_common transform's exact output on the attack-frame tensor (the native image), including the deployed pipeline's truncating uint8 round-trips (phase3_common.tensor_to_pil); those round-trips are straight-through in the backward pass. The attack output is converted to uint8 by truncation, the Phase-3 conversion: every perturbed pixel is floored, on average 0.5/255 below rounding.
- EOT loss 'mean' averages the 3 branch objectives; it ignores that the NMS-union merge keeps a box if ANY branch keeps it, so a box one branch still detects is not targeted specifically.
- EOT aggregates over deterministic defense branches, not over defense randomness (EnsJMG has none); a randomized defense would need EOT samples per branch as well.
- PGD init: one seeded uniform(-eps, eps) draw per (seed, image id, restart), identical for the oblivious and the adaptive attack, so the two differ only in the gradient they follow. Phase 3 drew its init from the unseeded global RNG, so Phase-3 and Phase-4 attacked numbers also differ by init noise.
- PGD iterate selection 'best': per restart, the last iterate and the lowest-objective iterate seen are both quantized and the one with the lower objective after quantization is kept (Phase 3 kept the last iterate). Applied to the oblivious and the adaptive attack alike.

## Timing

- **15.27 s/image** on `cuda:0` — 254.6 min for 1000 of 1000 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 1.27 h · n=1000: 4.24 h · n=5000: 21.21 h
