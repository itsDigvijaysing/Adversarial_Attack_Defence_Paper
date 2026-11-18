# Adaptive white-box: BPDA vs Median (google/owlvit-base-patch32)

- Run `20261118-105111` · model `owl` · experiment `adaptive_bpda`
- Images: **n = 1000** (`--select head`, seed 1, id digest `620b79f17bd75ab3`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Target **Median** (`median`) = median
- PGD eps=0.03 in pixel space at native resolution (uint8 output clipped to +/- 7/255), 50 steps, alpha=0.002, seed 1, 1 restart(s), iterate selection best; ensemble objective `max`

| Condition | mAP |
|---|---|
| clean_mAP | 0.3099 |
| clean_defended_mAP | 0.3043 |
| oblivious_attacked_undefended_mAP | 0.0108 |
| oblivious_attacked_defended_mAP | 0.1687 |
| adaptive_attacked_undefended_mAP | 0.2444 |
| adaptive_attacked_defended_mAP | 0.0213 |

- Images where the adaptive J is below the oblivious J: **1.000** of 1000
- BPDA: median: true a.e. gradient of the unfold median (each output routes to the input pixel its window selected)

## Timing

- **14.00 s/image** on `cuda:0` — 233.3 min for 1000 of 1000 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 1.17 h · n=1000: 3.89 h · n=5000: 19.44 h
