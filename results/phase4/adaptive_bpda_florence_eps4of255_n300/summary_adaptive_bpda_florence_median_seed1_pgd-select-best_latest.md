# Adaptive white-box: BPDA vs Median (microsoft/Florence-2-base, <OD>)

- Run `20261118-102046` · model `florence` · experiment `adaptive_bpda`
- Images: **n = 300** (`--select head`, seed 1, id digest `94c8dc9df724d3df`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Target **Median** (`median`) = median
- PGD eps=0.016 (pixel space), 50 iters, alpha=0.00107, 1 restart(s), seed 1; ensemble objective `max`; iterate selection `best`

## Headline (COCO mAP@[.5:.95], geometric pseudo-scores)

| Condition | mAP |
|---|---|
| clean, undefended | 0.3910 |
| clean, Median | 0.3896 |
| oblivious attack, undefended | 0.1954 |
| oblivious attack, Median | 0.3271 |
| adaptive attack, undefended | 0.3809 |
| adaptive attack, Median | 0.2142 |

- Images where the adaptive J is below the oblivious J: **1.000** of 300

## All conditions

| Tag | mAP | AP50 |
|---|---|---|
| `clean` | 0.3910 | 0.5090 |
| `clean+median` | 0.3896 | 0.5146 |
| `pgd_adaptive` | 0.3809 | 0.4983 |
| `pgd_adaptive+median` | 0.2142 | 0.3300 |
| `pgd_oblivious` | 0.1954 | 0.3148 |
| `pgd_oblivious+median` | 0.3271 | 0.4645 |

## Implementation notes

- Pixel space: the perturbation is added to the native-resolution image in [0,1] and the uint8 output is clipped to the clean image +/- 4/255. Phase 3 attacked the normalized 768x768 tensor (eps=0.03 there is about 1.7/255), so these numbers are not comparable with the Phase-3 Florence runs.
- The attacker's preprocessing is a differentiable stand-in for the processor (antialiased bicubic resize, normalize); max difference 0.1423 in normalized units on the first image. Scored inference always uses the real processor.
- Objective: token-level cross-entropy against the model's own clean decode under teacher forcing; scored inference uses beam search, which the attack does not differentiate through.
- Detection scores are the geometric pseudo-scores of compute_score (Florence <OD> emits no confidences), as in every Florence table.
- Single transform: --eot-loss has no effect.
- BPDA: median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected). Every forward is the deployed transform's exact value.
- Half precision on GPU, as in every Florence run; non-finite gradient entries are zeroed.

## Timing

- **58.84 s/image** on `cuda:0` — 294.2 min for 300 of 300 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 4.90 h · n=1000: 16.34 h · n=5000: 81.72 h
