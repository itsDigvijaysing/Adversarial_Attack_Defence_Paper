# Adaptive white-box: BPDA vs EnsJMG (google/owlvit-base-patch32)

- Run `20261118-105108` · model `owl` · experiment `adaptive_bpda`
- Images: **n = 1000** (`--select head`, seed 1, id digest `620b79f17bd75ab3`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Target **EnsJMG** (`ens_jpeg_median_gaussian`) = jpeg + median + gaussian
- PGD eps=0.03 in pixel space at native resolution (uint8 output clipped to +/- 7/255), 50 steps, alpha=0.002, seed 1, 1 restart(s), iterate selection best; ensemble objective `max`

| Condition | mAP |
|---|---|
| clean_mAP | 0.3099 |
| clean_defended_mAP | 0.3150 |
| oblivious_attacked_undefended_mAP | 0.0114 |
| oblivious_attacked_defended_mAP | 0.2010 |
| adaptive_attacked_undefended_mAP | 0.0408 |
| adaptive_attacked_defended_mAP | 0.0610 |

- Images where the adaptive J is below the oblivious J: **1.000** of 1000
- BPDA: jpeg: identity (BPDA straight-through); median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected); gaussian: true gradient

## Timing

- **18.41 s/image** on `cuda:0` — 306.9 min for 1000 of 1000 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 1.53 h · n=1000: 5.11 h · n=5000: 25.57 h
