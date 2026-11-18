# Adaptive white-box: BPDA vs EnsJMG (google/owlvit-base-patch32)

- Run `20261118-105109` · model `owl` · experiment `adaptive_bpda`
- Images: **n = 300** (`--select head`, seed 1, id digest `94c8dc9df724d3df`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Target **EnsJMG** (`ens_jpeg_median_gaussian`) = jpeg + median + gaussian
- PGD eps=0.03 in pixel space at native resolution (uint8 output clipped to +/- 7/255), 50 steps, alpha=0.002, seed 1, 1 restart(s), iterate selection best; ensemble objective `mean`

| Condition | mAP |
|---|---|
| clean_mAP | 0.3231 |
| clean_defended_mAP | 0.3371 |
| oblivious_attacked_undefended_mAP | 0.0123 |
| oblivious_attacked_defended_mAP | 0.2232 |
| adaptive_attacked_undefended_mAP | 0.0459 |
| adaptive_attacked_defended_mAP | 0.0728 |

- Images where the adaptive J is below the oblivious J: **1.000** of 300
- BPDA: jpeg: identity (BPDA straight-through); median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected); gaussian: true gradient

## Timing

- **29.45 s/image** on `cuda:0` — 147.3 min for 300 of 300 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 2.45 h · n=1000: 8.18 h · n=5000: 40.91 h
