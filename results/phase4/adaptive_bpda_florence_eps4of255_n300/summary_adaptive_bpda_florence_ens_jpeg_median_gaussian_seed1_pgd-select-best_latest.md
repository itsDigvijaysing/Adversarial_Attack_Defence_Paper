# Adaptive white-box: BPDA vs EnsJMG (microsoft/Florence-2-base, <OD>)

- Run `20261118-102042` · model `florence` · experiment `adaptive_bpda`
- Images: **n = 300** (`--select head`, seed 1, id digest `94c8dc9df724d3df`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Target **EnsJMG** (`ens_jpeg_median_gaussian`) = jpeg + median + gaussian
- PGD eps=0.016 (pixel space), 50 iters, alpha=0.00107, 1 restart(s), seed 1; ensemble objective `max`; iterate selection `best`

## Headline (COCO mAP@[.5:.95], geometric pseudo-scores)

| Condition | mAP |
|---|---|
| clean, undefended | 0.3910 |
| clean, EnsJMG | 0.3860 |
| oblivious attack, undefended | 0.1941 |
| oblivious attack, EnsJMG | 0.3323 |
| adaptive attack, undefended | 0.2319 |
| adaptive attack, EnsJMG | 0.2555 |

- Images where the adaptive J is below the oblivious J: **1.000** of 300

## All conditions

| Tag | mAP | AP50 |
|---|---|---|
| `clean` | 0.3910 | 0.5090 |
| `clean+ens_jpeg_median_gaussian` | 0.3860 | 0.5209 |
| `clean+gaussian` | 0.3302 | 0.4305 |
| `clean+jpeg` | 0.3828 | 0.5022 |
| `clean+median` | 0.3896 | 0.5146 |
| `pgd_adaptive` | 0.2319 | 0.3359 |
| `pgd_adaptive+ens_jpeg_median_gaussian` | 0.2555 | 0.3739 |
| `pgd_adaptive+gaussian` | 0.2235 | 0.3156 |
| `pgd_adaptive+jpeg` | 0.2326 | 0.3316 |
| `pgd_adaptive+median` | 0.2492 | 0.3588 |
| `pgd_oblivious` | 0.1941 | 0.3134 |
| `pgd_oblivious+ens_jpeg_median_gaussian` | 0.3323 | 0.4763 |
| `pgd_oblivious+gaussian` | 0.3149 | 0.4256 |
| `pgd_oblivious+jpeg` | 0.3342 | 0.4523 |
| `pgd_oblivious+median` | 0.3422 | 0.4793 |

## Implementation notes

- Pixel space: the perturbation is added to the native-resolution image in [0,1] and the uint8 output is clipped to the clean image +/- 4/255. Phase 3 attacked the normalized 768x768 tensor (eps=0.03 there is about 1.7/255), so these numbers are not comparable with the Phase-3 Florence runs.
- The attacker's preprocessing is a differentiable stand-in for the processor (antialiased bicubic resize, normalize); max difference 0.1423 in normalized units on the first image. Scored inference always uses the real processor.
- Objective: token-level cross-entropy against the model's own clean decode under teacher forcing; scored inference uses beam search, which the attack does not differentiate through.
- Detection scores are the geometric pseudo-scores of compute_score (Florence <OD> emits no confidences), as in every Florence table.
- Ensemble objective 'max': each step raises the cross-entropy of the least damaged branch. A proxy for the NMS merge, which keeps a box if any branch keeps it.
- BPDA: jpeg: identity (BPDA straight-through); median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected); gaussian: true gradient. Every forward is the deployed transform's exact value.
- Half precision on GPU, as in every Florence run; non-finite gradient entries are zeroed.

## Timing

- **77.37 s/image** on `cuda:0` — 386.9 min for 300 of 300 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 6.45 h · n=1000: 21.49 h · n=5000: 107.46 h
