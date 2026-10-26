# Phase-3 v2 detection dumps, re-scored on CPU (`rescore_phase3_v2`)

- Run `20261024-012907`, git `5112d30-dirty`. No model and no image was loaded: inputs are the COCO-format dumps the Phase-3 v2 runs wrote, plus the COCO GT.
- Images: **n = 1000**, the first 1000 sorted val2017 ids (the Phase-3 subset), id digest `620b79f17bd75ab3`.
- Bootstrap: **B = 1000** paired image-level resamples (seed 42, one resample matrix shared by every condition), 95% percentile CIs, engine `fast`.
- summary.json agreement (|diff| <= 0.0001): **114/114** values match, max |diff| = 0.00e+00.
- Fast accumulate vs pycocotools: max |diff| 0.0e+00 on the full set, 0.0e+00 over 216 duplicated-id resamples.
- Clean dumps byte-identical across the attack dirs: YOLOv8x-worldv2 yes, Florence-2-base yes.
- Image ids in the YOLOv8x-worldv2 dumps: 1000 distinct over 36 dumps, all inside the scored set (0 detections outside); images without a detection still count (their GT are misses).
- Image ids in the Florence-2-base dumps: 981 distinct over 36 dumps, all inside the scored set (0 detections outside); images without a detection still count (their GT are misses).

## How to read

- Every number is mAP@[.5:.95] unless the column says AP50 or F1. Brackets are the bootstrap CI; "(CI spans 0)" marks a difference the 1000-image sample cannot separate from zero.
- **Clean cost** = mAP(defense on clean images) - mAP(clean). Negative = the defense costs accuracy when there is no attack.
- **Recovery** = mAP(defended) - mAP(attacked), same images in each resample. "share <= 0" is the fraction of resamples in which the defense did not help; it is not a p-value.
- The CIs cover image sampling only. Each condition is still one attack run (no seeds), so they do not capture attack or model randomness.
- Florence-2 boxes carry geometric pseudo-scores (box size and centring, not confidence), so its mAP ranks boxes by geometry. The score-free table uses no score beyond tie-breaking.
- Best ensemble is chosen on the point estimate and held fixed in the bootstrap, which favours it slightly.

## Clean cost per defense

### YOLOv8x-worldv2 (clean mAP 0.4765, AP50 0.6400)

| Defense | Kind | mAP on clean | Clean cost, mAP [CI] | Clean cost, AP50 [CI] |
|---|---|---|---|---|
| JPEG | solo | 0.4525 | -0.0240 [-0.0333, -0.0146] | -0.0244 [-0.0346, -0.0140] |
| Median | solo | 0.4583 | -0.0181 [-0.0255, -0.0099] | -0.0189 [-0.0262, -0.0091] |
| TVM | solo | 0.4472 | -0.0292 [-0.0379, -0.0214] | -0.0309 [-0.0410, -0.0210] |
| Gaussian | solo | 0.4614 | -0.0151 [-0.0241, -0.0066] | -0.0129 [-0.0234, -0.0021] |
| Blur+TVM | solo | 0.4149 | -0.0616 [-0.0734, -0.0504] | -0.0683 [-0.0812, -0.0532] |
| EnsBTC | ensemble | 0.4647 | -0.0118 [-0.0197, -0.0036] | -0.0099 [-0.0195, -0.0005] |
| EnsJMG | ensemble | 0.4656 | -0.0109 [-0.0175, -0.0023] | -0.0074 [-0.0140, +0.0022] (CI spans 0) (BC CI [-0.0194, -0.0016] excludes 0) |
| EnsJMT | ensemble | 0.4671 | -0.0094 [-0.0151, -0.0019] | -0.0080 [-0.0140, +0.0005] (CI spans 0) (BC CI [-0.0183, -0.0026] excludes 0) |

### Florence-2-base (clean mAP 0.3605, AP50 0.4876)

| Defense | Kind | mAP on clean | Clean cost, mAP [CI] | Clean cost, AP50 [CI] |
|---|---|---|---|---|
| JPEG | solo | 0.3523 | -0.0083 [-0.0211, +0.0040] (CI spans 0) | -0.0163 [-0.0321, -0.0010] |
| Median | solo | 0.3528 | -0.0077 [-0.0207, +0.0070] (CI spans 0) | -0.0133 [-0.0292, +0.0038] (CI spans 0) |
| TVM | solo | 0.2891 | -0.0715 [-0.0876, -0.0601] | -0.0985 [-0.1186, -0.0835] |
| Gaussian | solo | 0.2914 | -0.0691 [-0.0852, -0.0562] | -0.0959 [-0.1152, -0.0783] |
| Blur+TVM | solo | 0.2652 | -0.0954 [-0.1149, -0.0809] | -0.1290 [-0.1529, -0.1078] |
| EnsBTC | ensemble | 0.3520 | -0.0086 [-0.0199, +0.0085] (CI spans 0) | +0.0011 [-0.0117, +0.0203] (CI spans 0) |
| EnsJMG | ensemble | 0.3513 | -0.0092 [-0.0204, +0.0080] (CI spans 0) | -0.0014 [-0.0135, +0.0191] (CI spans 0) |
| EnsJMT | ensemble | 0.3563 | -0.0043 [-0.0151, +0.0115] (CI spans 0) | +0.0047 [-0.0079, +0.0237] (CI spans 0) |

## Recovery under attack (defended - attacked)

### YOLOv8x-worldv2 x FGSM (eps 0.03)

| Defense | Kind | mAP | Recovery, mAP [CI] | Recovery, AP50 [CI] | share <= 0 |
|---|---|---|---|---|---|
| no defense (attacked) | | 0.2259 | damage vs clean: -0.2506 [-0.2705, -0.2348] | -0.3031 [-0.3229, -0.2802] | |
| JPEG | solo | 0.2697 | +0.0438 [+0.0371, +0.0517] | +0.0566 [+0.0465, +0.0658] | 0.000 |
| Median | solo | 0.2928 | +0.0669 [+0.0596, +0.0759] | +0.0885 [+0.0767, +0.0988] | 0.000 |
| TVM | solo | 0.2966 | +0.0707 [+0.0613, +0.0823] | +0.0930 [+0.0794, +0.1056] | 0.000 |
| Gaussian | solo | 0.3171 | +0.0911 [+0.0825, +0.1032] | +0.1172 [+0.1037, +0.1312] | 0.000 |
| Blur+TVM | solo | 0.3448 | +0.1189 [+0.1054, +0.1334] | +0.1550 [+0.1367, +0.1709] | 0.000 |
| EnsBTC | ensemble | 0.3364 | +0.1104 [+0.0996, +0.1234] | +0.1422 [+0.1272, +0.1562] | 0.000 |
| EnsJMG | ensemble | 0.3098 | +0.0839 [+0.0763, +0.0955] | +0.1075 [+0.0968, +0.1203] | 0.000 |
| EnsJMT | ensemble | 0.3058 | +0.0799 [+0.0721, +0.0918] | +0.1043 [+0.0932, +0.1167] | 0.000 |

### YOLOv8x-worldv2 x PGD (eps 0.03)

| Defense | Kind | mAP | Recovery, mAP [CI] | Recovery, AP50 [CI] | share <= 0 |
|---|---|---|---|---|---|
| no defense (attacked) | | 0.0737 | damage vs clean: -0.4028 [-0.4313, -0.3863] | -0.5178 [-0.5474, -0.4946] | |
| JPEG | solo | 0.3643 | +0.2905 [+0.2732, +0.3149] | +0.3902 [+0.3673, +0.4174] | 0.000 |
| Median | solo | 0.3639 | +0.2902 [+0.2786, +0.3149] | +0.3895 [+0.3717, +0.4186] | 0.000 |
| TVM | solo | 0.4049 | +0.3312 [+0.3172, +0.3568] | +0.4440 [+0.4221, +0.4714] | 0.000 |
| Gaussian | solo | 0.4053 | +0.3315 [+0.3176, +0.3568] | +0.4396 [+0.4197, +0.4678] | 0.000 |
| Blur+TVM | solo | 0.4025 | +0.3288 [+0.3142, +0.3545] | +0.4324 [+0.4110, +0.4617] | 0.000 |
| EnsBTC | ensemble | 0.4128 | +0.3391 [+0.3263, +0.3662] | +0.4465 [+0.4264, +0.4767] | 0.000 |
| EnsJMG | ensemble | 0.4045 | +0.3308 [+0.3178, +0.3580] | +0.4404 [+0.4202, +0.4700] | 0.000 |
| EnsJMT | ensemble | 0.4114 | +0.3377 [+0.3242, +0.3643] | +0.4474 [+0.4265, +0.4760] | 0.000 |

### YOLOv8x-worldv2 x Patch

| Defense | Kind | mAP | Recovery, mAP [CI] | Recovery, AP50 [CI] | share <= 0 |
|---|---|---|---|---|---|
| no defense (attacked) | | 0.4537 | damage vs clean: -0.0228 [-0.0315, -0.0158] | -0.0233 [-0.0324, -0.0153] | |
| JPEG | solo | 0.4391 | -0.0146 [-0.0224, -0.0064] | -0.0172 [-0.0255, -0.0070] | 1.000 |
| Median | solo | 0.4496 | -0.0040 [-0.0116, +0.0086] (CI spans 0) | -0.0057 [-0.0133, +0.0079] (CI spans 0) | 0.603 |
| TVM | solo | 0.4288 | -0.0249 [-0.0337, -0.0165] | -0.0275 [-0.0381, -0.0181] | 1.000 |
| Gaussian | solo | 0.4511 | -0.0026 [-0.0103, +0.0076] (CI spans 0) | -0.0009 [-0.0101, +0.0104] (CI spans 0) | 0.606 |
| Blur+TVM | solo | 0.4093 | -0.0444 [-0.0555, -0.0336] | -0.0534 [-0.0655, -0.0396] | 1.000 |
| EnsBTC | ensemble | 0.4563 | +0.0027 [-0.0042, +0.0119] (CI spans 0) | +0.0024 [-0.0055, +0.0128] (CI spans 0) | 0.167 |
| EnsJMG | ensemble | 0.4574 | +0.0037 [-0.0039, +0.0162] (CI spans 0) | +0.0053 [-0.0021, +0.0185] (CI spans 0) | 0.108 |
| EnsJMT | ensemble | 0.4599 | +0.0063 [-0.0007, +0.0148] (CI spans 0) | +0.0057 [-0.0018, +0.0151] (CI spans 0) | 0.034 |

### Florence-2-base x FGSM (eps 0.03)

| Defense | Kind | mAP | Recovery, mAP [CI] | Recovery, AP50 [CI] | share <= 0 |
|---|---|---|---|---|---|
| no defense (attacked) | | 0.2731 | damage vs clean: -0.0874 [-0.1075, -0.0711] | -0.0976 [-0.1216, -0.0742] | |
| JPEG | solo | 0.3199 | +0.0468 [+0.0358, +0.0593] | +0.0519 [+0.0324, +0.0711] | 0.000 |
| Median | solo | 0.3194 | +0.0463 [+0.0351, +0.0609] | +0.0506 [+0.0351, +0.0684] | 0.000 |
| TVM | solo | 0.2631 | -0.0100 [-0.0278, +0.0049] (CI spans 0) | -0.0308 [-0.0534, -0.0121] | 0.911 |
| Gaussian | solo | 0.2848 | +0.0117 [-0.0033, +0.0273] (CI spans 0) | +0.0028 [-0.0199, +0.0248] (CI spans 0) | 0.075 |
| Blur+TVM | solo | 0.2745 | +0.0014 [-0.0164, +0.0157] (CI spans 0) | -0.0185 [-0.0434, +0.0001] (CI spans 0) | 0.483 |
| EnsBTC | ensemble | 0.3262 | +0.0531 [+0.0418, +0.0703] | +0.0708 [+0.0521, +0.0930] | 0.000 |
| EnsJMG | ensemble | 0.3281 | +0.0550 [+0.0435, +0.0717] | +0.0723 [+0.0538, +0.0941] | 0.000 |
| EnsJMT | ensemble | 0.3261 | +0.0530 [+0.0426, +0.0694] | +0.0715 [+0.0524, +0.0929] | 0.000 |

### Florence-2-base x PGD (eps 0.03)

| Defense | Kind | mAP | Recovery, mAP [CI] | Recovery, AP50 [CI] | share <= 0 |
|---|---|---|---|---|---|
| no defense (attacked) | | 0.2243 | damage vs clean: -0.1363 [-0.1597, -0.1188] | -0.1597 [-0.1908, -0.1321] | |
| JPEG | solo | 0.3162 | +0.0919 [+0.0797, +0.1106] | +0.1028 [+0.0853, +0.1269] | 0.000 |
| Median | solo | 0.3198 | +0.0955 [+0.0810, +0.1122] | +0.1098 [+0.0878, +0.1304] | 0.000 |
| TVM | solo | 0.2622 | +0.0379 [+0.0202, +0.0532] | +0.0306 [+0.0057, +0.0510] | 0.000 |
| Gaussian | solo | 0.2874 | +0.0631 [+0.0476, +0.0804] | +0.0627 [+0.0380, +0.0848] | 0.000 |
| Blur+TVM | solo | 0.2723 | +0.0480 [+0.0290, +0.0660] | +0.0396 [+0.0141, +0.0639] | 0.000 |
| EnsBTC | ensemble | 0.3243 | +0.1001 [+0.0866, +0.1204] | +0.1278 [+0.1076, +0.1538] | 0.000 |
| EnsJMG | ensemble | 0.3245 | +0.1002 [+0.0865, +0.1202] | +0.1271 [+0.1067, +0.1532] | 0.000 |
| EnsJMT | ensemble | 0.3228 | +0.0985 [+0.0843, +0.1190] | +0.1254 [+0.1037, +0.1518] | 0.000 |

### Florence-2-base x Patch

| Defense | Kind | mAP | Recovery, mAP [CI] | Recovery, AP50 [CI] | share <= 0 |
|---|---|---|---|---|---|
| no defense (attacked) | | 0.3268 | damage vs clean: -0.0338 [-0.0460, -0.0234] | -0.0482 [-0.0648, -0.0351] | |
| JPEG | solo | 0.3270 | +0.0002 [-0.0085, +0.0128] (CI spans 0) | -0.0036 [-0.0140, +0.0159] (CI spans 0) | 0.363 |
| Median | solo | 0.3289 | +0.0021 [-0.0064, +0.0152] (CI spans 0) | -0.0009 [-0.0120, +0.0205] (CI spans 0) | 0.231 |
| TVM | solo | 0.2652 | -0.0615 [-0.0773, -0.0496] | -0.0812 [-0.1001, -0.0663] | 1.000 |
| Gaussian | solo | 0.2903 | -0.0365 [-0.0497, -0.0247] | -0.0535 [-0.0708, -0.0352] | 1.000 |
| Blur+TVM | solo | 0.2676 | -0.0591 [-0.0758, -0.0425] | -0.0771 [-0.0954, -0.0560] | 1.000 |
| EnsBTC | ensemble | 0.3324 | +0.0057 [-0.0040, +0.0242] (CI spans 0) | +0.0188 [+0.0083, +0.0422] | 0.081 |
| EnsJMG | ensemble | 0.3290 | +0.0022 [-0.0042, +0.0170] (CI spans 0) | +0.0095 [+0.0008, +0.0342] (BC CI [-0.0058, +0.0199] spans 0) | 0.138 |
| EnsJMT | ensemble | 0.3307 | +0.0040 [-0.0040, +0.0176] (CI spans 0) | +0.0139 [+0.0051, +0.0345] (BC CI [-0.0017, +0.0233] spans 0) | 0.085 |

## Best ensemble vs Median, under attack

The paper's headline said only ensembles are positive in every setting; Median is also positive in every setting of its own tables. This is the paired difference between the two.

| Model | Attack | Best ensemble | Ensemble mAP | Median mAP | Ensemble - Median [CI] | share <= 0 |
|---|---|---|---|---|---|---|
| YOLOv8x-worldv2 | FGSM (eps 0.03) | EnsBTC | 0.3364 | 0.2928 | +0.0435 [+0.0335, +0.0540] | 0.000 |
| YOLOv8x-worldv2 | PGD (eps 0.03) | EnsBTC | 0.4128 | 0.3639 | +0.0489 [+0.0386, +0.0587] | 0.000 |
| YOLOv8x-worldv2 | Patch | EnsJMT | 0.4599 | 0.4496 | +0.0103 [+0.0017, +0.0161] | 0.005 |
| Florence-2-base | FGSM (eps 0.03) | EnsJMG | 0.3281 | 0.3194 | +0.0087 [+0.0003, +0.0185] (BC CI [-0.0020, +0.0167] spans 0) | 0.023 |
| Florence-2-base | PGD (eps 0.03) | EnsJMG | 0.3245 | 0.3198 | +0.0047 [-0.0008, +0.0150] (CI spans 0) | 0.043 |
| Florence-2-base | Patch | EnsBTC | 0.3324 | 0.3289 | +0.0036 [-0.0056, +0.0181] (CI spans 0) | 0.186 |

Every ensemble - Median (mAP [CI]), so the choice of the best one does not drive the verdict:

| Model | Attack | EnsBTC | EnsJMG | EnsJMT |
|---|---|---|---|---|
| YOLOv8x-worldv2 | FGSM (eps 0.03) | +0.0435 [+0.0335, +0.0540] | +0.0170 [+0.0117, +0.0269] | +0.0130 [+0.0058, +0.0237] |
| YOLOv8x-worldv2 | PGD (eps 0.03) | +0.0489 [+0.0386, +0.0587] | +0.0406 [+0.0322, +0.0492] | +0.0474 [+0.0373, +0.0569] |
| YOLOv8x-worldv2 | Patch | +0.0067 [-0.0007, +0.0122] (CI spans 0) (BC CI [+0.0016, +0.0158] excludes 0) | +0.0077 [+0.0038, +0.0120] | +0.0103 [+0.0017, +0.0161] |
| Florence-2-base | FGSM (eps 0.03) | +0.0069 [-0.0013, +0.0174] (CI spans 0) | +0.0087 [+0.0003, +0.0185] (BC CI [-0.0020, +0.0167] spans 0) | +0.0068 [-0.0013, +0.0161] (CI spans 0) |
| Florence-2-base | PGD (eps 0.03) | +0.0045 [-0.0007, +0.0153] (CI spans 0) | +0.0047 [-0.0008, +0.0150] (CI spans 0) | +0.0030 [-0.0032, +0.0132] (CI spans 0) |
| Florence-2-base | Patch | +0.0036 [-0.0056, +0.0181] (CI spans 0) | +0.0001 [-0.0047, +0.0085] (CI spans 0) | +0.0019 [-0.0055, +0.0117] (CI spans 0) |

## Score-free detection quality (IoU >= 0.5)

One-to-one greedy matching per image and class, pairs ordered by IoU (score breaks ties only), crowd regions ignored, counts pooled over images and classes. Cells: precision / recall / F1; the F1 CI is the paired bootstrap of the same B resamples.

For YOLO read the conf >= 0.25 table first: at the dump threshold (0.001) most boxes are low-confidence ones that mAP ranks last but a count metric scores as false positives, so ensembles (which merge three branches' boxes) look worse there by construction.

### YOLOv8x-worldv2, all dumped boxes (conf >= 0.001)

| Defense | Clean P / R / F1 | FGSM P / R / F1 | PGD P / R / F1 | Patch P / R / F1 |
|---|---|---|---|---|
| none | 0.087 / 0.860 / 0.158 | 0.064 / 0.639 / 0.116 | 0.066 / 0.341 / 0.111 | 0.084 / 0.852 / 0.153 |
| JPEG | 0.095 / 0.832 / 0.171 | 0.087 / 0.674 / 0.154 | 0.117 / 0.746 / 0.202 | 0.092 / 0.825 / 0.166 |
| Median | 0.089 / 0.844 / 0.161 | 0.087 / 0.710 / 0.155 | 0.111 / 0.767 / 0.194 | 0.086 / 0.839 / 0.156 |
| TVM | 0.079 / 0.846 / 0.145 | 0.073 / 0.731 / 0.133 | 0.089 / 0.816 / 0.161 | 0.076 / 0.834 / 0.140 |
| Gaussian | 0.103 / 0.836 / 0.183 | 0.101 / 0.721 / 0.176 | 0.119 / 0.786 / 0.207 | 0.099 / 0.831 / 0.178 |
| Blur+TVM | 0.084 / 0.800 / 0.152 | 0.084 / 0.743 / 0.151 | 0.089 / 0.784 / 0.160 | 0.081 / 0.795 / 0.147 |
| EnsBTC | 0.056 / 0.874 / 0.105 | 0.053 / 0.793 / 0.099 | 0.063 / 0.842 / 0.117 | 0.054 / 0.870 / 0.102 |
| EnsJMG | 0.064 / 0.871 / 0.119 | 0.060 / 0.764 / 0.112 | 0.077 / 0.830 / 0.141 | 0.062 / 0.867 / 0.115 |
| EnsJMT | 0.056 / 0.882 / 0.105 | 0.051 / 0.780 / 0.096 | 0.065 / 0.852 / 0.120 | 0.054 / 0.876 / 0.101 |

| Defense | Clean cost, F1 [CI] | FGSM recovery, F1 [CI] | PGD recovery, F1 [CI] | Patch recovery, F1 [CI] |
|---|---|---|---|---|
| JPEG | +0.013 [+0.011, +0.015] | +0.038 [+0.035, +0.040] | +0.091 [+0.084, +0.098] | +0.013 [+0.010, +0.015] |
| Median | +0.003 [+0.001, +0.004] | +0.039 [+0.036, +0.042] | +0.083 [+0.076, +0.091] | +0.003 [+0.001, +0.005] |
| TVM | -0.013 [-0.015, -0.011] | +0.017 [+0.014, +0.019] | +0.050 [+0.043, +0.057] | -0.014 [-0.016, -0.012] |
| Gaussian | +0.025 [+0.022, +0.028] | +0.060 [+0.057, +0.064] | +0.096 [+0.089, +0.104] | +0.024 [+0.022, +0.027] |
| Blur+TVM | -0.006 [-0.009, -0.003] | +0.035 [+0.032, +0.039] | +0.049 [+0.042, +0.056] | -0.007 [-0.009, -0.004] |
| EnsBTC | -0.053 [-0.056, -0.050] | -0.017 [-0.020, -0.014] | +0.006 [+0.000, +0.013] | -0.052 [-0.055, -0.049] |
| EnsJMG | -0.039 [-0.041, -0.037] | -0.004 [-0.007, -0.002] | +0.030 [+0.024, +0.036] | -0.038 [-0.040, -0.036] |
| EnsJMT | -0.053 [-0.055, -0.050] | -0.020 [-0.022, -0.017] | +0.009 [+0.003, +0.015] | -0.052 [-0.055, -0.050] |

### YOLOv8x-worldv2, conf >= 0.25

| Defense | Clean P / R / F1 | FGSM P / R / F1 | PGD P / R / F1 | Patch P / R / F1 |
|---|---|---|---|---|
| none | 0.693 / 0.643 / 0.667 | 0.638 / 0.369 / 0.468 | 0.626 / 0.101 / 0.174 | 0.698 / 0.599 / 0.645 |
| JPEG | 0.735 / 0.528 / 0.615 | 0.732 / 0.354 / 0.477 | 0.823 / 0.362 / 0.503 | 0.733 / 0.511 / 0.602 |
| Median | 0.712 / 0.596 / 0.649 | 0.719 / 0.411 / 0.523 | 0.802 / 0.406 / 0.539 | 0.709 / 0.588 / 0.643 |
| TVM | 0.695 / 0.603 / 0.646 | 0.708 / 0.420 / 0.527 | 0.777 / 0.483 / 0.596 | 0.698 / 0.571 / 0.628 |
| Gaussian | 0.740 / 0.565 / 0.641 | 0.740 / 0.415 / 0.532 | 0.803 / 0.454 / 0.580 | 0.734 / 0.558 / 0.634 |
| Blur+TVM | 0.736 / 0.517 / 0.607 | 0.759 / 0.432 / 0.551 | 0.766 / 0.469 / 0.582 | 0.727 / 0.510 / 0.599 |
| EnsBTC | 0.648 / 0.627 / 0.638 | 0.654 / 0.473 / 0.549 | 0.715 / 0.507 / 0.593 | 0.641 / 0.615 / 0.628 |
| EnsJMG | 0.665 / 0.627 / 0.646 | 0.663 / 0.446 / 0.533 | 0.752 / 0.482 / 0.587 | 0.661 / 0.616 / 0.638 |
| EnsJMT | 0.641 / 0.649 / 0.645 | 0.643 / 0.454 / 0.532 | 0.728 / 0.513 / 0.602 | 0.638 / 0.631 / 0.635 |

| Defense | Clean cost, F1 [CI] | FGSM recovery, F1 [CI] | PGD recovery, F1 [CI] | Patch recovery, F1 [CI] |
|---|---|---|---|---|
| JPEG | -0.053 [-0.065, -0.041] | +0.010 [-0.002, +0.021] (CI spans 0) | +0.329 [+0.311, +0.346] | -0.043 [-0.054, -0.031] |
| Median | -0.018 [-0.026, -0.011] | +0.055 [+0.047, +0.063] | +0.365 [+0.347, +0.383] | -0.002 [-0.010, +0.005] (CI spans 0) |
| TVM | -0.021 [-0.028, -0.014] | +0.059 [+0.051, +0.068] | +0.422 [+0.403, +0.440] | -0.017 [-0.024, -0.010] |
| Gaussian | -0.027 [-0.037, -0.017] | +0.064 [+0.056, +0.074] | +0.406 [+0.388, +0.423] | -0.011 [-0.020, -0.002] |
| Blur+TVM | -0.060 [-0.071, -0.049] | +0.083 [+0.071, +0.095] | +0.408 [+0.388, +0.426] | -0.046 [-0.057, -0.035] |
| EnsBTC | -0.030 [-0.036, -0.024] | +0.081 [+0.072, +0.091] | +0.419 [+0.401, +0.437] | -0.017 [-0.024, -0.011] |
| EnsJMG | -0.022 [-0.028, -0.015] | +0.065 [+0.057, +0.074] | +0.413 [+0.395, +0.431] | -0.007 [-0.014, -0.001] |
| EnsJMT | -0.023 [-0.029, -0.017] | +0.064 [+0.055, +0.073] | +0.428 [+0.410, +0.445] | -0.010 [-0.016, -0.004] |

### Florence-2-base, all predicted boxes

| Defense | Clean P / R / F1 | FGSM P / R / F1 | PGD P / R / F1 | Patch P / R / F1 |
|---|---|---|---|---|
| none | 0.710 / 0.581 / 0.639 | 0.667 / 0.498 / 0.570 | 0.723 / 0.421 / 0.532 | 0.741 / 0.515 / 0.608 |
| JPEG | 0.711 / 0.555 / 0.624 | 0.689 / 0.540 / 0.605 | 0.712 / 0.527 / 0.606 | 0.715 / 0.537 / 0.613 |
| Median | 0.713 / 0.554 / 0.624 | 0.686 / 0.538 / 0.603 | 0.723 / 0.521 / 0.606 | 0.718 / 0.527 / 0.608 |
| TVM | 0.813 / 0.434 / 0.566 | 0.812 / 0.412 / 0.546 | 0.817 / 0.404 / 0.541 | 0.816 / 0.411 / 0.546 |
| Gaussian | 0.774 / 0.438 / 0.559 | 0.742 / 0.440 / 0.552 | 0.765 / 0.435 / 0.555 | 0.756 / 0.441 / 0.557 |
| Blur+TVM | 0.795 / 0.404 / 0.535 | 0.792 / 0.403 / 0.534 | 0.794 / 0.400 / 0.532 | 0.783 / 0.407 / 0.536 |
| EnsBTC | 0.656 / 0.605 / 0.629 | 0.626 / 0.590 / 0.608 | 0.659 / 0.574 / 0.613 | 0.658 / 0.576 / 0.615 |
| EnsJMG | 0.659 / 0.604 / 0.630 | 0.630 / 0.589 / 0.609 | 0.660 / 0.574 / 0.614 | 0.660 / 0.576 / 0.615 |
| EnsJMT | 0.662 / 0.606 / 0.632 | 0.631 / 0.591 / 0.610 | 0.661 / 0.573 / 0.614 | 0.665 / 0.575 / 0.617 |

| Defense | Clean cost, F1 [CI] | FGSM recovery, F1 [CI] | PGD recovery, F1 [CI] | Patch recovery, F1 [CI] |
|---|---|---|---|---|
| JPEG | -0.015 [-0.025, -0.006] | +0.035 [+0.021, +0.048] | +0.074 [+0.059, +0.089] | +0.005 [-0.003, +0.014] (CI spans 0) |
| Median | -0.015 [-0.024, -0.005] | +0.033 [+0.020, +0.047] | +0.074 [+0.059, +0.090] | +0.000 [-0.011, +0.013] (CI spans 0) |
| TVM | -0.073 [-0.088, -0.058] | -0.024 [-0.037, -0.010] | +0.009 [-0.007, +0.023] (CI spans 0) | -0.061 [-0.075, -0.048] |
| Gaussian | -0.080 [-0.095, -0.066] | -0.018 [-0.034, -0.002] | +0.023 [+0.006, +0.040] | -0.050 [-0.065, -0.036] |
| Blur+TVM | -0.103 [-0.120, -0.088] | -0.036 [-0.052, -0.020] | -0.000 [-0.018, +0.018] (CI spans 0) | -0.072 [-0.086, -0.057] |
| EnsBTC | -0.009 [-0.018, -0.001] | +0.038 [+0.023, +0.052] | +0.081 [+0.067, +0.097] | +0.007 [-0.004, +0.018] (CI spans 0) |
| EnsJMG | -0.009 [-0.017, +0.000] (CI spans 0) (BC CI [-0.018, -0.000] excludes 0) | +0.039 [+0.024, +0.053] | +0.082 [+0.067, +0.097] | +0.008 [-0.003, +0.019] (CI spans 0) |
| EnsJMT | -0.006 [-0.015, +0.003] (CI spans 0) | +0.040 [+0.026, +0.055] | +0.082 [+0.067, +0.098] | +0.009 [-0.001, +0.020] (CI spans 0) |

## Bootstrap sanity

- Bootstrap mean vs point estimate (mAP), over 72 conditions: median |bias| 0.0107, max |bias| 0.0151 (1.42 SE, `florence/patch+ens_blur_tvm_combo`). Typical SE 0.0101.
- Flagged (|bias| > 0.5 SE): all 72 conditions.
- A single condition's bootstrap mAP sits above its point estimate. Likely reasons (inferred, not isolated here): the interpolated precision envelope is a running maximum, and the maximum of a noisier curve is larger on average; classes absent from a resample drop out of the class mean. Both act on the two terms of a paired difference alike, so read the difference CIs, not CIs of single mAPs.
- Paired mAP differences (88): median |bias| 0.25 SE, max 0.75 SE (`yolo/pgd/recovery/median`, 0.0071). "Excludes 0" verdicts that change under a bias-corrected (BC) percentile CI: **2** (`florence/fgsm/ens_jpeg_median_gaussian - median`, `yolo/patch/ens_blur_tvm_combo - median`). Such cells carry the BC interval inline.
- Paired AP50 differences (76): median |bias| 0.19 SE, max 0.75 SE (`florence/patch/recovery/ens_jpeg_median_gaussian`, 0.0062). "Excludes 0" verdicts that change under a bias-corrected (BC) percentile CI: **4** (`florence/patch/recovery/ens_jpeg_median_gaussian`, `florence/patch/recovery/ens_jpeg_median_tvm`, `yolo/clean cost/ens_jpeg_median_gaussian`, `yolo/clean cost/ens_jpeg_median_tvm`). Such cells carry the BC interval inline.
