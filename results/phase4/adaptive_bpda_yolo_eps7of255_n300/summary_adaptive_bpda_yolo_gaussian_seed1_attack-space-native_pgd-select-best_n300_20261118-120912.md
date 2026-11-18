# Adaptive white-box pilot: BPDA + EOT vs Gaussian (yolov8x-worldv2.pt)

- Run `20261118-120912` · model `yolo` · experiment `adaptive_bpda`
- Images: **n = 300** (`--select head`, seed 1, id digest `94c8dc9df724d3df`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Target **Gaussian** (`gaussian`) = gaussian, solo transform (no merge)
- PGD eps=0.03, 50 iters, alpha=0.00200, 1 restart(s), raw [0,1] space; seeded init per (seed 1, image id, restart), shared by both attacks
- Attack space `native` · output quantization `truncate` · median surrogate `true` · iterate selection `best`
- EOT loss `max`: single branch: EOT is trivial and --eot-loss has no effect
- Execution: EOT batch `on` (auto: 31.5 GiB >= 16 GiB) (an execution choice: same objective, equal up to float rounding)
- BPDA surrogates (backward only; every forward is the deployed value): gaussian: true gradient
- All headline numbers come from the SAME image subset.

## Headline (COCO mAP@[.5:.95])

| Condition | mAP |
|---|---|
| clean, undefended | 0.5032 |
| clean, Gaussian | 0.4935 |
| oblivious attack, undefended | 0.0175 |
| oblivious attack, Gaussian | 0.4430 |
| adaptive attack, undefended | 0.2860 |
| adaptive attack, Gaussian | 0.0351 |

- Defense recovery vs an oblivious attacker: **+0.4255**
- Defense recovery vs an adaptive attacker: **-0.2509**
- Defended mAP lost to adaptivity: **-0.4079**

## Surrogate objective (the attacker's own view)

J = sum over anchors of the max-class confidence, measured THROUGH Gaussian with the attack's own surrogate (deployed forward values, native frame, eot-loss max); lower = more suppression. A diagnostic of the attacker's optimisation, not a detection metric.

| Input | mean J through the target |
|---|---|
| clean | 39.3057 |
| oblivious adversarial | 27.2013 |
| adaptive adversarial | 0.4923 |

- Images where the adaptive J is below the oblivious J: **1.000** of 300 (should be close to 1; if not, the adaptive attack is not optimising its objective)

## All conditions

| Input | Defense | Tag | mAP | AP50 |
|---|---|---|---|---|
| clean | undefended | `clean` | 0.5032 | 0.6732 |
| clean | Gaussian | `clean+gaussian` | 0.4935 | 0.6641 |
| adaptive attack | undefended | `pgd_adaptive` | 0.2860 | 0.4093 |
| adaptive attack | Gaussian | `pgd_adaptive+gaussian` | 0.0351 | 0.0578 |
| oblivious attack | undefended | `pgd_oblivious` | 0.0175 | 0.0322 |
| oblivious attack | Gaussian | `pgd_oblivious+gaussian` | 0.4430 | 0.6108 |

## Implementation uncertainty

- Native space: the perturbation lives at native resolution, each surrogate defense runs there as deployed, and the model input is a differentiable copy of ultralytics' predict letterbox (minimum-rectangle padding to the stride, grey 114, centred). It resizes with torch bilinear where ultralytics uses cv2 INTER_LINEAR on uint8, so the two agree exactly when no resize is needed (longest side == 640) and differ by interpolation rounding otherwise (under 1/255 per pixel against ultralytics' LetterBox in tests/test_attack_units.py). The letterbox-vs-native scale mismatch of the default mode is removed and nothing resamples the perturbation after the attack. The uint8 output is clipped to the clean image +/- 7/255 (eps*255 = 7.65 is not whole, and truncation or rounding alone would move a pixel by 8/255), so no pixel the defense receives moves more than eps.
- True gradients: gaussian: true gradient. The median gradient is sparse (one input per output pixel), so it carries less signal per step than a smooth filter's.
- Every surrogate forwards the deployed phase3_common transform's exact output on the attack-frame tensor (the native image), including the deployed pipeline's truncating uint8 round-trips (phase3_common.tensor_to_pil); those round-trips are straight-through in the backward pass. The attack output is converted to uint8 by truncation, the Phase-3 conversion: every perturbed pixel is floored, on average 0.5/255 below rounding.
- EOT aggregates over deterministic defense branches, not over defense randomness (Gaussian has none); a randomized defense would need EOT samples per branch as well.
- PGD init: one seeded uniform(-eps, eps) draw per (seed, image id, restart), identical for the oblivious and the adaptive attack, so the two differ only in the gradient they follow. Phase 3 drew its init from the unseeded global RNG, so Phase-3 and Phase-4 attacked numbers also differ by init noise.
- PGD iterate selection 'best': per restart, the last iterate and the lowest-objective iterate seen are both quantized and the one with the lower objective after quantization is kept (Phase 3 kept the last iterate). Applied to the oblivious and the adaptive attack alike.

## Timing

- **19.11 s/image** on `cuda:0` — 95.5 min for 300 of 300 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 1.59 h · n=1000: 5.31 h · n=5000: 26.54 h
