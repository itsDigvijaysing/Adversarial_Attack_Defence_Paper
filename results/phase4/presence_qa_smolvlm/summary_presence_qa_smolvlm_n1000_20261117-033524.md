# Object-presence questions: HuggingFaceTB/SmolVLM-256M-Instruct

- Run `20261117-033524` · model `smolvlm` · experiment `presence_qa`
- Images: **n = 1000** (`--select head`, seed 1, id digest `620b79f17bd75ab3`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- 2000 yes/no questions (2 per image), accuracy against the COCO annotations; chance is 0.5 on balanced questions
- Oblivious PGD eps=0.03, 10 iters, alpha=0.00750, on the 512x512 square
- Brackets: 95% paired bootstrap intervals over images (B=1000)

- **Clean accuracy 0.8560 · attacked 0.4385** (damage +0.4175)

| Defense | Kind | Clean acc. | Clean cost | Attacked acc. | Recovery |
|---|---|---|---|---|---|
| JPEG (`jpeg`) | solo | 0.8265 | -0.029 [-0.041, -0.018] | 0.4825 | +0.044 [+0.034, +0.054] |
| Median (`median`) | solo | 0.8445 | -0.011 [-0.022, -0.001] | 0.5310 | +0.092 [+0.081, +0.106] |
| Gaussian (`gaussian`) | solo | 0.8195 | -0.036 [-0.049, -0.024] | 0.5800 | +0.141 [+0.128, +0.159] |
| TVM (`tvm`) | solo | 0.8315 | -0.025 [-0.037, -0.013] | 0.6380 | +0.200 [+0.182, +0.217] |
| Blur+TVM (`blur_tvm`) | solo | 0.7860 | -0.070 [-0.084, -0.057] | 0.7535 | +0.315 [+0.297, +0.335] |
| Bilateral (`bilateral`) | solo | 0.8420 | -0.014 [-0.024, -0.004] | 0.4525 | +0.014 [+0.006, +0.021] |
| SVD (ensemble member) (`svd`) | solo | 0.8550 | -0.001 [-0.004, +0.001] | 0.4385 | +0.000 [+0.000, +0.000] |
| Bit-depth (`bit_depth`) | solo | 0.8500 | -0.006 [-0.016, +0.004] | 0.4380 | -0.001 [-0.004, +0.003] |
| Ordered dithering (`dithering`) | solo | 0.8525 | -0.004 [-0.012, +0.005] | 0.4390 | +0.001 [-0.004, +0.004] |
| Non-local means (`nlm`) | solo | 0.8525 | -0.004 [-0.009, +0.002] | 0.4410 | +0.003 [-0.003, +0.009] |
| Anisotropic diffusion (`anisotropic`) | solo | 0.8455 | -0.011 [-0.021, -0.001] | 0.4645 | +0.026 [+0.017, +0.034] |
| BM3D (`bm3d`) | solo | 0.8100 | -0.046 [-0.059, -0.033] | 0.7280 | +0.289 [+0.272, +0.308] |
| Random resize-and-pad (`random_resize`) | solo | 0.8250 | -0.031 [-0.043, -0.019] | 0.7475 | +0.309 [+0.291, +0.328] |
| EnsJMG (`ens_jpeg_median_gaussian`) | ensemble | 0.8340 | -0.022 [-0.032, -0.011] | 0.5225 | +0.084 [+0.072, +0.097] |
| EnsJMT (`ens_jpeg_median_tvm`) | ensemble | 0.8385 | -0.018 [-0.028, -0.007] | 0.5215 | +0.083 [+0.071, +0.096] |
| Ens4W (`ens_4way`) | ensemble | 0.8350 | -0.021 [-0.031, -0.011] | 0.5315 | +0.093 [+0.081, +0.105] |
| EnsBTC (`ens_blur_tvm_combo`) | ensemble | 0.8305 | -0.025 [-0.036, -0.015] | 0.5380 | +0.100 [+0.087, +0.113] |
| EnsBMT (`ens_bilateral_median`) | ensemble | 0.8425 | -0.013 [-0.023, -0.004] | 0.5070 | +0.069 [+0.057, +0.081] |
| EnsJBT (`ens_jpeg_bilateral_tvm`) | ensemble | 0.8370 | -0.019 [-0.030, -0.009] | 0.4785 | +0.040 [+0.030, +0.050] |
| EnsJTS (`ens_jpeg_tvm_svd`) | ensemble | 0.8460 | -0.010 [-0.018, -0.002] | 0.4700 | +0.032 [+0.024, +0.041] |

## Timing

- **26.71 s/image** on `cuda:0` — 445.2 min for 1000 of 1000 images computed this run (resumed images excluded from the rate)
- extrapolated: n=1000: 7.42 h · n=4000: 29.68 h · n=5000: 37.10 h
