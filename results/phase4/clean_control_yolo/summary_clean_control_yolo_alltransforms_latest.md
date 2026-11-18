# Clean-image control — yolov8x-worldv2.pt (defenses on UNATTACKED images)

- Run `20261117-033609` · model `yolo` · experiment `clean_control`
- Images: **n = 5000** (`--select head`, seed 1, id digest `e69f60365d651b46`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Metric per track: `yolo` = COCO mAP@[.5:.95]
- A small negative delta is expected — it is the clean-accuracy cost of the transform, and it is what tells a reviewer whether a defense counters the attack or just changes the image.

## Undefended clean baseline measured by this run

| Track | measured | published N=5000 (input) | difference |
|---|---|---|---|
| yolo | 0.4519 | 0.4519 | +0.0000 |

> The published column is an INPUT, not a measurement: Florence-2 detection comes from the committed N=5000 summaries, YOLO and OCR from the N=5000 runs done on teammates' machines. A different n or subset makes the difference a subset effect, not a discrepancy.

## Clean-image performance per defense

Δ is versus **this run's own** undefended clean number (same images, same code path).

| Defense | Kind | yolo | Δ |
|---|---|---|---|
| JPEG (`jpeg`) | solo | 0.4262 | -0.0257 |
| Median (`median`) | solo | 0.4387 | -0.0132 |
| Gaussian (`gaussian`) | solo | 0.4369 | -0.0150 |
| TVM (`tvm`) | solo | 0.4280 | -0.0239 |
| Blur+TVM (`blur_tvm`) | solo | 0.3953 | -0.0566 |
| Bilateral (`bilateral`) | solo | 0.4463 | -0.0056 |
| SVD (ensemble member) (`svd`) | solo | 0.4519 | +0.0000 |
| Bit-depth (`bit_depth`) | solo | 0.4448 | -0.0071 |
| Ordered dithering (`dithering`) | solo | 0.4431 | -0.0088 |
| Non-local means (`nlm`) | solo | 0.4503 | -0.0017 |
| Anisotropic diffusion (`anisotropic`) | solo | 0.4399 | -0.0120 |
| EnsJMG (`ens_jpeg_median_gaussian`) | ENSEMBLE | 0.4468 | -0.0052 |
| EnsJMT (`ens_jpeg_median_tvm`) | ENSEMBLE | 0.4466 | -0.0053 |
| Ens4W (`ens_4way`) | ENSEMBLE | 0.4480 | -0.0039 |
| EnsBTC (`ens_blur_tvm_combo`) | ENSEMBLE | 0.4424 | -0.0095 |
| EnsBMT (`ens_bilateral_median`) | ENSEMBLE | 0.4477 | -0.0042 |
| EnsJBT (`ens_jpeg_bilateral_tvm`) | ENSEMBLE | 0.4480 | -0.0039 |
| EnsJTS (`ens_jpeg_tvm_svd`) | ENSEMBLE | 0.4522 | +0.0003 |

## Timing

s/image is measured over the images actually computed this run; checkpoint-resumed images are excluded, so a resumed run still reports a real rate. Device: `cuda:0`.

| Track | images | computed now | s/image | this run | est. n=1000 | est. n=5000 |
|---|---|---|---|---|---|---|
| yolo | 5000 | 5000 | 1.77 | 147.3 min | 0.49 h | 2.45 h |
