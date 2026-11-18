# Adaptive white-box: BPDA vs Median (microsoft/Florence-2-base, <OD>)

- Run `20261117-033439` · model `florence` · experiment `adaptive_bpda`
- Images: **n = 1000** (`--select head`, seed 1, id digest `620b79f17bd75ab3`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Target **Median** (`median`) = median
- PGD eps=0.03 (pixel space), 50 iters, alpha=0.00200, 1 restart(s), seed 1; ensemble objective `max`; iterate selection `best`

## Headline (COCO mAP@[.5:.95], geometric pseudo-scores)

| Condition | mAP |
|---|---|
| clean, undefended | 0.3604 |
| clean, Median | 0.3504 |
| oblivious attack, undefended | 0.1381 |
| oblivious attack, Median | 0.2561 |
| adaptive attack, undefended | 0.3346 |
| adaptive attack, Median | 0.1480 |

- Images where the adaptive J is below the oblivious J: **1.000** of 1000

## All conditions

| Tag | mAP | AP50 |
|---|---|---|
| `clean` | 0.3604 | 0.4873 |
| `clean+median` | 0.3504 | 0.4677 |
| `pgd_adaptive` | 0.3346 | 0.4685 |
| `pgd_adaptive+median` | 0.1480 | 0.2451 |
| `pgd_oblivious` | 0.1381 | 0.2359 |
| `pgd_oblivious+median` | 0.2561 | 0.3850 |

## Implementation notes

- Pixel space: the perturbation is added to the native-resolution image in [0,1] and the uint8 output is clipped to the clean image +/- 7/255. Phase 3 attacked the normalized 768x768 tensor (eps=0.03 there is about 1.7/255), so these numbers are not comparable with the Phase-3 Florence runs.
- The attacker's preprocessing is a differentiable stand-in for the processor (antialiased bicubic resize, normalize); max difference 0.1423 in normalized units on the first image. Scored inference always uses the real processor.
- Objective: token-level cross-entropy against the model's own clean decode under teacher forcing; scored inference uses beam search, which the attack does not differentiate through.
- Detection scores are the geometric pseudo-scores of compute_score (Florence <OD> emits no confidences), as in every Florence table.
- Single transform: --eot-loss has no effect.
- BPDA: median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected). Every forward is the deployed transform's exact value.
- Half precision on GPU, as in every Florence run; non-finite gradient entries are zeroed.

## Timing

- **45.49 s/image** on `cuda:0` — 758.2 min for 1000 of 1000 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 3.79 h · n=1000: 12.64 h · n=5000: 63.18 h
