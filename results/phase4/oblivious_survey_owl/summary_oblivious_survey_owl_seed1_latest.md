# Oblivious survey: every transform on google/owlvit-base-patch32

- Run `20261118-105114` · model `owl` · experiment `oblivious_survey`
- Images: **n = 1000** (`--select head`, seed 1, id digest `620b79f17bd75ab3`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- PGD eps=0.03 in pixel space at native resolution (uint8 output clipped to +/- 7/255), 50 steps, alpha=0.002, seed 1, 1 restart(s), iterate selection best; the attacker ignores the defense
- COCO mAP@[.5:.95] with OWL-ViT's own confidences

Clean **0.3099**, attacked undefended **0.0117** (damage 0.2982). Best single transform under attack: Blur+TVM 0.2362; best ensemble: EnsBTC 0.2434.

| Defense | Kind | Clean mAP | Clean cost | Attacked mAP | Recovery | % of damage |
|---|---|---|---|---|---|---|
| EnsBTC | ensemble | 0.3115 | +0.0016 | 0.2434 | +0.2317 | 77.7 |
| Ens4W | ensemble | 0.3144 | +0.0045 | 0.2366 | +0.2250 | 75.4 |
| Blur+TVM | solo | 0.2680 | -0.0418 | 0.2362 | +0.2245 | 75.3 |
| EnsJMT | ensemble | 0.3130 | +0.0032 | 0.2332 | +0.2215 | 74.3 |
| EnsJBT | ensemble | 0.3137 | +0.0038 | 0.2325 | +0.2208 | 74.1 |
| EnsJTS | ensemble | 0.3157 | +0.0058 | 0.2322 | +0.2205 | 74.0 |
| EnsBMT | ensemble | 0.3126 | +0.0028 | 0.2313 | +0.2196 | 73.7 |
| TVM | solo | 0.2945 | -0.0153 | 0.2288 | +0.2171 | 72.8 |
| EnsJMG | ensemble | 0.3150 | +0.0052 | 0.2006 | +0.1889 | 63.4 |
| Gaussian | solo | 0.2969 | -0.0129 | 0.1920 | +0.1803 | 60.5 |
| Median | solo | 0.3043 | -0.0056 | 0.1684 | +0.1568 | 52.6 |
| JPEG | solo | 0.3037 | -0.0062 | 0.1673 | +0.1556 | 52.2 |
| Anisotropic diffusion | solo | 0.3048 | -0.0051 | 0.1455 | +0.1339 | 44.9 |
| Bilateral | solo | 0.3058 | -0.0040 | 0.1389 | +0.1272 | 42.7 |
| Non-local means | solo | 0.3096 | -0.0003 | 0.1144 | +0.1027 | 34.4 |
| Bit-depth | solo | 0.3007 | -0.0092 | 0.0497 | +0.0380 | 12.8 |
| Ordered dithering | solo | 0.2993 | -0.0106 | 0.0471 | +0.0354 | 11.9 |
| SVD (ensemble member) | solo | 0.3112 | +0.0014 | 0.0124 | +0.0007 | 0.2 |

## Timing

- **11.80 s/image** on `cuda:0` — 196.7 min for 1000 of 1000 images computed this run (resumed images excluded from the rate)
- extrapolated: n=300: 0.98 h · n=1000: 3.28 h · n=5000: 16.39 h
