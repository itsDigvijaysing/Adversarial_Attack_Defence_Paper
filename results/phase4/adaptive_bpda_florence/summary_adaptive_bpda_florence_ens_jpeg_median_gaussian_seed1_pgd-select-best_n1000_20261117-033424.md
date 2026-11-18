# Adaptive white-box: BPDA vs EnsJMG (microsoft/Florence-2-base, <OD>)

- Run `20261117-033424` · model `florence` · experiment `adaptive_bpda`
- Images: **n = 1000** (`--select head`, seed 1, id digest `620b79f17bd75ab3`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Target **EnsJMG** (`ens_jpeg_median_gaussian`) = jpeg + median + gaussian
- PGD eps=0.03 (pixel space), 50 iters, alpha=0.00200, 1 restart(s), seed 1; ensemble objective `max`; iterate selection `best`

## Headline (COCO mAP@[.5:.95], geometric pseudo-scores)

| Condition | mAP |
|---|---|
| clean, undefended | 0.3604 |
| clean, EnsJMG | 0.3529 |
| oblivious attack, undefended | 0.1336 |
| oblivious attack, EnsJMG | 0.2642 |
| adaptive attack, undefended | 0.1588 |
| adaptive attack, EnsJMG | 0.1656 |

- Images where the adaptive J is below the oblivious J: **1.000** of 1000

## All conditions

| Tag | mAP | AP50 |
|---|---|---|
| `clean` | 0.3604 | 0.4873 |
| `clean+ens_jpeg_median_gaussian` | 0.3529 | 0.4880 |
| `clean+gaussian` | 0.2930 | 0.3945 |
| `clean+jpeg` | 0.3527 | 0.4711 |
| `clean+median` | 0.3504 | 0.4677 |
| `pgd_adaptive` | 0.1588 | 0.2597 |
| `pgd_adaptive+ens_jpeg_median_gaussian` | 0.1656 | 0.2708 |
| `pgd_adaptive+gaussian` | 0.1446 | 0.2366 |
| `pgd_adaptive+jpeg` | 0.1616 | 0.2579 |
| `pgd_adaptive+median` | 0.1529 | 0.2509 |
| `pgd_oblivious` | 0.1336 | 0.2234 |
| `pgd_oblivious+ens_jpeg_median_gaussian` | 0.2642 | 0.4008 |
| `pgd_oblivious+gaussian` | 0.2452 | 0.3624 |
| `pgd_oblivious+jpeg` | 0.2595 | 0.3811 |
| `pgd_oblivious+median` | 0.2495 | 0.3728 |

## Implementation notes

- Pixel space: the perturbation is added to the native-resolution image in [0,1] and the uint8 output is clipped to the clean image +/- 7/255. Phase 3 attacked the normalized 768x768 tensor (eps=0.03 there is about 1.7/255), so these numbers are not comparable with the Phase-3 Florence runs.
- The attacker's preprocessing is a differentiable stand-in for the processor (antialiased bicubic resize, normalize); max difference 0.1423 in normalized units on the first image. Scored inference always uses the real processor.
- Objective: token-level cross-entropy against the model's own clean decode under teacher forcing; scored inference uses beam search, which the attack does not differentiate through.
- Detection scores are the geometric pseudo-scores of compute_score (Florence <OD> emits no confidences), as in every Florence table.
- Ensemble objective 'max': each step raises the cross-entropy of the least damaged branch. A proxy for the NMS merge, which keeps a box if any branch keeps it.
- BPDA: jpeg: identity (BPDA straight-through); median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected); gaussian: true gradient. Every forward is the deployed transform's exact value.
- Half precision on GPU, as in every Florence run; non-finite gradient entries are zeroed.

## Timing

- **67.55 s/image** on `cuda:0` — 1125.8 min for 1000 of 1000 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 5.63 h · n=1000: 18.76 h · n=5000: 93.82 h
