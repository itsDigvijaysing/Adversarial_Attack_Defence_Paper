# Clean-image control — microsoft/Florence-2-base (defenses on UNATTACKED images)

- Run `20261117-033454` · model `florence` · experiment `clean_control`
- Images: **n = 5000** (`--select head`, seed 1, id digest `e69f60365d651b46`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Metric per track: `florence_det` = COCO mAP@[.5:.95]
- A small negative delta is expected — it is the clean-accuracy cost of the transform, and it is what tells a reviewer whether a defense counters the attack or just changes the image.

## Undefended clean baseline measured by this run

| Track | measured | published N=5000 (input) | difference |
|---|---|---|---|
| florence_det | 0.3304 | 0.3300 | +0.0004 |

> The published column is an INPUT, not a measurement: Florence-2 detection comes from the committed N=5000 summaries, YOLO and OCR from the N=5000 runs done on teammates' machines. A different n or subset makes the difference a subset effect, not a discrepancy.

## Clean-image performance per defense

Δ is versus **this run's own** undefended clean number (same images, same code path).

| Defense | Kind | florence_det | Δ |
|---|---|---|---|
| JPEG (`jpeg`) | solo | 0.3198 | -0.0106 |
| Median (`median`) | solo | 0.3202 | -0.0102 |
| Gaussian (`gaussian`) | solo | 0.2744 | -0.0560 |
| TVM (`tvm`) | solo | 0.2643 | -0.0662 |
| Blur+TVM (`blur_tvm`) | solo | 0.2492 | -0.0812 |
| Bilateral (`bilateral`) | solo | 0.3314 | +0.0010 |
| SVD (ensemble member) (`svd`) | solo | 0.3284 | -0.0020 |
| Bit-depth (`bit_depth`) | solo | 0.2802 | -0.0502 |
| Ordered dithering (`dithering`) | solo | 0.3036 | -0.0269 |
| Non-local means (`nlm`) | solo | 0.3107 | -0.0198 |
| Anisotropic diffusion (`anisotropic`) | solo | 0.3163 | -0.0141 |
| EnsJMG (`ens_jpeg_median_gaussian`) | ENSEMBLE | 0.3242 | -0.0062 |
| EnsJMT (`ens_jpeg_median_tvm`) | ENSEMBLE | 0.3241 | -0.0063 |
| Ens4W (`ens_4way`) | ENSEMBLE | 0.3212 | -0.0092 |
| EnsBTC (`ens_blur_tvm_combo`) | ENSEMBLE | 0.3202 | -0.0103 |
| EnsBMT (`ens_bilateral_median`) | ENSEMBLE | 0.3258 | -0.0046 |
| EnsJBT (`ens_jpeg_bilateral_tvm`) | ENSEMBLE | 0.3253 | -0.0052 |
| EnsJTS (`ens_jpeg_tvm_svd`) | ENSEMBLE | 0.3238 | -0.0066 |

## Timing

s/image is measured over the images actually computed this run; checkpoint-resumed images are excluded, so a resumed run still reports a real rate. Device: `cuda:0`.

| Track | images | computed now | s/image | this run | est. n=1000 | est. n=5000 |
|---|---|---|---|---|---|---|
| florence_det | 5000 | 5000 | 16.93 | 1411.2 min | 4.70 h | 23.52 h |
