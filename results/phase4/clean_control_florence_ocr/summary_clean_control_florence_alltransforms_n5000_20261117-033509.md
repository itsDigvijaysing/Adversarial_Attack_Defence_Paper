# Clean-image control — microsoft/Florence-2-base (defenses on UNATTACKED images)

- Run `20261117-033509` · model `florence` · experiment `clean_control`
- Images: **n = 5000** (`--select head`, seed 1, id digest `e69f60365d651b46`)
- Image dir `/home/manu/m_work/Adversarial_Attack_Defence_Paper/val2017` · annotations `/home/manu/m_work/Adversarial_Attack_Defence_Paper/annotations/instances_val2017.json`
- Metric per track: `florence_ocr` = OCR self-consistency similarity
- A small negative delta is expected — it is the clean-accuracy cost of the transform, and it is what tells a reviewer whether a defense counters the attack or just changes the image.

## Undefended clean baseline measured by this run

| Track | measured | published N=5000 (input) | difference |
|---|---|---|---|
| florence_ocr | 1.0000 | 1.0000 | +0.0000 |

> The published column is an INPUT, not a measurement: Florence-2 detection comes from the committed N=5000 summaries, YOLO and OCR from the N=5000 runs done on teammates' machines. A different n or subset makes the difference a subset effect, not a discrepancy.

## Clean-image performance per defense

Δ is versus **this run's own** undefended clean number (same images, same code path).

| Defense | Kind | florence_ocr | Δ |
|---|---|---|---|
| JPEG (`jpeg`) | solo | 0.6734 | -0.3266 |
| Median (`median`) | solo | 0.6203 | -0.3797 |
| Gaussian (`gaussian`) | solo | 0.6715 | -0.3285 |
| TVM (`tvm`) | solo | 0.7247 | -0.2753 |
| Blur+TVM (`blur_tvm`) | solo | 0.5069 | -0.4931 |
| Bilateral (`bilateral`) | solo | 0.8210 | -0.1790 |
| SVD (ensemble member) (`svd`) | solo | 0.9531 | -0.0469 |
| Bit-depth (`bit_depth`) | solo | 0.7966 | -0.2034 |
| Ordered dithering (`dithering`) | solo | 0.7761 | -0.2239 |
| Non-local means (`nlm`) | solo | 0.8856 | -0.1144 |
| Anisotropic diffusion (`anisotropic`) | solo | 0.7976 | -0.2024 |
| BM3D (`bm3d`) | solo | 0.6477 | -0.3523 |
| Random resize-and-pad (`random_resize`) | solo | 0.7070 | -0.2930 |
| EnsJMG (`ens_jpeg_median_gaussian`) | ENSEMBLE | 0.7132 | -0.2868 |
| EnsJMT (`ens_jpeg_median_tvm`) | ENSEMBLE | 0.7462 | -0.2538 |
| Ens4W (`ens_4way`) | ENSEMBLE | 0.7455 | -0.2545 |
| EnsBTC (`ens_blur_tvm_combo`) | ENSEMBLE | 0.6442 | -0.3558 |
| EnsBMT (`ens_bilateral_median`) | ENSEMBLE | 0.7882 | -0.2118 |
| EnsJBT (`ens_jpeg_bilateral_tvm`) | ENSEMBLE | 0.7888 | -0.2112 |
| EnsJTS (`ens_jpeg_tvm_svd`) | ENSEMBLE | 0.8427 | -0.1573 |

## Timing

s/image is measured over the images actually computed this run; checkpoint-resumed images are excluded, so a resumed run still reports a real rate. Device: `cuda:0`.

| Track | images | computed now | s/image | this run | est. n=1000 | est. n=5000 |
|---|---|---|---|---|---|---|
| florence_ocr | 5000 | 5000 | 14.83 | 1236.1 min | 4.12 h | 20.60 h |
