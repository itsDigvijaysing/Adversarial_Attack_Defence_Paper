# # Adversarial Patch Phase 3 — YOLOv8x-worldv2
# 
# **Attack:** 35x35 center-placed patch, 100 Adam optimization steps.
# **Defenses:** same 5 solos + top-3 ensembles as FGSM/PGD notebooks.
# 
# 3000 images from COCO val2017 | single GPU | checkpoint every 1000 images.
# 

# ## 1. Imports & GPU isolation

import os
import sys
import json
import time
import pickle
import warnings

# ----- GPU isolation (must happen BEFORE importing torch) -----
NUM_GPUS = 1
import subprocess
try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    gpu_free = []
    for line in smi.stdout.strip().split("\n"):
        idx, free = line.split(",")
        gpu_free.append((int(idx.strip()), int(free.strip())))
    gpu_free.sort(key=lambda x: x[1], reverse=True)
    selected = [str(g[0]) for g in gpu_free[:NUM_GPUS]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
except Exception as e:
    print(f"[GPU] nvidia-smi failed ({e}), using default CUDA_VISIBLE_DEVICES")

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from pycocotools.coco import COCO

from ultralytics import YOLO

import phase3_common as pc

warnings.filterwarnings("ignore")
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | GPUs visible: {torch.cuda.device_count()}")


# ## 2. GPU diagnostics

assert torch.cuda.is_available(), "CUDA required for Phase 3."
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info(0)
print(f"Device: {device} -> {props.name}")
print(f"Memory: {free/1024**3:.2f} GB free / {total/1024**3:.2f} GB total")


# ## 3. Configuration

# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_DIR = "./Dataset/val2017"
ANN_FILE  = "./Dataset/annotations/instances_val2017.json"

NUM_IMAGES    = 3000
CHECKPOINT_EVERY = 1000

# Attack config
ATTACK_TAG = "patch"
EPSILON    = 0.03         # FGSM / PGD perturbation budget
PGD_ITERS  = 10
PGD_ALPHA  = EPSILON / 4.0

# Patch config (only used by patch notebook)
PATCH_SIZE      = pc.PATCH_SIZE        # 35
PATCH_OPT_ITERS = pc.PATCH_OPT_ITERS   # 100
PATCH_LR        = pc.PATCH_LR          # 0.02

# Defense hyperparams (matched to VariantZ winners)
JPEG_QUALITY       = 75
MEDIAN_KERNEL      = 3
TVM_WEIGHT         = 0.05
TVM_ITERS          = 200
GAUSSIAN_SIGMA     = 1.0
ENSEMBLE_NMS_IOU   = 0.5

# YOLO settings
YOLO_MODEL   = "yolov8x-worldv2.pt"
YOLO_IMGSZ   = 640
YOLO_CONF    = 0.001
YOLO_IOU_NMS = 0.5

SOLO_DEFENSES = pc.SOLO_DEFENSES   # ["jpeg", "median", "tvm", "gaussian", "blur_tvm"]
ENSEMBLES     = pc.ENSEMBLES       # top-3 from VariantZ

OUTPUT_DIR      = "./results_phase3_yolo_patch"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "detections.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)


pc.print_banner(f"Phase 3 YOLO — {ATTACK_TAG}", width=70)
print(f"Images: {NUM_IMAGES} | Attack: {ATTACK_TAG} | eps={EPSILON}")
print(f"Solo defenses ({len(SOLO_DEFENSES)}): {SOLO_DEFENSES}")
print(f"Ensembles     ({len(ENSEMBLES)}):")
for n, m in ENSEMBLES.items():
    print(f"  {n:<30s} = {' + '.join(m)}")
print(f"Output: {OUTPUT_DIR}")


# ## 4. Load YOLOv8x-worldv2 + COCO GT

# ============================================================
# Load YOLOv8x-worldv2 + COCO GT
# ============================================================
coco_gt = COCO(ANN_FILE)
cats_sorted = sorted(coco_gt.loadCats(coco_gt.getCatIds()), key=lambda x: x["id"])
COCO_NAMES = [c["name"] for c in cats_sorted]
COCO_IDS   = [c["id"]   for c in cats_sorted]
YOLO_TO_COCO_ID = {i: cid for i, cid in enumerate(COCO_IDS)}

model = YOLO(YOLO_MODEL)
model.set_classes(COCO_NAMES)
model.to(device)
model.model.eval()
print(f"YOLO loaded — {len(COCO_NAMES)} COCO classes | imgsz={YOLO_IMGSZ}")

files = sorted(os.listdir(IMAGE_DIR))
if NUM_IMAGES is not None:
    files = files[:NUM_IMAGES]
evaluated_img_ids = sorted({int(os.path.splitext(f)[0]) for f in files})
print(f"Selected {len(files)} images.")


# ## 5. Inference utility

# ============================================================
# Inference (single image -> COCO-format detections)
# ============================================================
def _parse_yolo_result(res):
    dets = []
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return dets
    xyxy  = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss  = boxes.cls.cpu().numpy().astype(int)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy[i]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        cls_idx = int(clss[i])
        if cls_idx not in YOLO_TO_COCO_ID:
            continue
        dets.append({
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "category_id": YOLO_TO_COCO_ID[cls_idx],
            "score": float(confs[i]),
        })
    return dets

def run_inference(pil_img):
    results = model.predict(pil_img, conf=YOLO_CONF, iou=YOLO_IOU_NMS,
                            imgsz=YOLO_IMGSZ, verbose=False, device=device)
    return _parse_yolo_result(results[0])


# ## 6. Letterbox helpers

# ============================================================
# Letterbox helpers (shared by all attacks — match YOLO preprocessing)
# ============================================================
def letterbox(pil_img, size=YOLO_IMGSZ, fill=(114, 114, 114)):
    w, h = pil_img.size
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = pil_img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    pl, pt = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pl, pt))
    return canvas, scale, pl, pt, nw, nh

def unletterbox(lb_np, orig_size, pl, pt, nw, nh):
    cropped = lb_np[pt:pt + nh, pl:pl + nw]
    return Image.fromarray(cropped).resize(orig_size, Image.BICUBIC)


# ## 7. Patch Attack

# ============================================================
# Adversarial Patch — 35x35, center-placed, 100 optimization steps
# ============================================================
def patch_attack(pil_img, patch_size=PATCH_SIZE,
                 iters=PATCH_OPT_ITERS, lr=PATCH_LR):
    orig_size = pil_img.size
    lb, _, pl, pt, nw, nh = letterbox(pil_img, YOLO_IMGSZ)
    img_np = np.array(lb).astype(np.float32) / 255.0
    img_t  = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    H, W = img_t.shape[-2:]
    top, left = pc.center_patch_coords(H, W, patch_size)

    torch.manual_seed(0)
    patch = torch.rand(3, patch_size, patch_size, device=device) * 0.5 + 0.25
    patch.requires_grad_(True)
    optim = torch.optim.Adam([patch], lr=lr)

    for _ in range(iters):
        optim.zero_grad()
        composed = img_t.clone()
        composed[:, :, top:top + patch_size, left:left + patch_size] = (
            patch.clamp(0.0, 1.0).unsqueeze(0)
        )
        model.model.model[-1].shape = None  # reset cached inference tensors for autograd
        preds = model.model(composed)
        pred  = preds[0] if isinstance(preds, (list, tuple)) else preds
        cls_scores = pred[:, 4:, :]
        loss = cls_scores.max(dim=1)[0].sum()
        loss.backward()
        optim.step()
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)

    with torch.no_grad():
        adv = img_t.clone()
        adv[:, :, top:top + patch_size, left:left + patch_size] = (
            patch.clamp(0.0, 1.0).unsqueeze(0)
        )
    adv_np = (adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return unletterbox(adv_np, orig_size, pl, pt, nw, nh)

print(f"Patch attack ready ({PATCH_SIZE}x{PATCH_SIZE}, {PATCH_OPT_ITERS} iters).")


# ## 8. Bind `attack_fn` for the shared main loop

attack_fn = patch_attack
print(f'attack_fn bound to {attack_fn.__name__}')


# ## 9. Sanity check

# ============================================================
# Sanity check: one image end-to-end before launching the main loop
# ============================================================
_test_path = os.path.join(IMAGE_DIR, files[0])
_test_img  = Image.open(_test_path).convert("RGB")
_dets_clean = run_inference(_test_img)
print(f"  clean            -> {len(_dets_clean)} dets")
_adv = attack_fn(_test_img)
_dets_atk = run_inference(_adv)
print(f"  attacked         -> {len(_dets_atk)} dets")
_defs = pc.apply_all_defenses_gpu(
    _adv, device,
    jpeg_quality=JPEG_QUALITY, median_kernel=MEDIAN_KERNEL,
    tvm_weight=TVM_WEIGHT, tvm_iters=TVM_ITERS,
    gaussian_sigma=GAUSSIAN_SIGMA,
)
for dn, dimg in _defs.items():
    print(f"  attacked+{dn:<10s} -> {len(run_inference(dimg))} dets")
print("Sanity check PASSED.")


# ## 10. Main evaluation loop (checkpointed every 1000 images)

# ============================================================
# Main evaluation loop — single GPU, sequential per-image, checkpointed
# ============================================================
ckpt = pc.DetectionCheckpoint(CHECKPOINT_PATH, flush_every=CHECKPOINT_EVERY)

def process_one(fname):
    img_id = int(os.path.splitext(fname)[0])
    if ckpt.has(img_id):
        return
    pil_img = Image.open(os.path.join(IMAGE_DIR, fname)).convert("RGB")
    buckets = {}

    def _record(tag, dets):
        for d in dets:
            d["image_id"] = img_id
        buckets[tag] = dets

    # 1. Clean baseline
    _record("clean", run_inference(pil_img))
    # 2. Clean + solo defenses
    clean_defs = pc.apply_all_defenses_gpu(
        pil_img, device,
        jpeg_quality=JPEG_QUALITY, median_kernel=MEDIAN_KERNEL,
        tvm_weight=TVM_WEIGHT, tvm_iters=TVM_ITERS,
        gaussian_sigma=GAUSSIAN_SIGMA,
    )
    for dn in SOLO_DEFENSES:
        _record(f"clean+{dn}", run_inference(clean_defs[dn]))
    # 3. Attacked + solo defenses
    adv = attack_fn(pil_img)
    _record(ATTACK_TAG, run_inference(adv))
    atk_defs = pc.apply_all_defenses_gpu(
        adv, device,
        jpeg_quality=JPEG_QUALITY, median_kernel=MEDIAN_KERNEL,
        tvm_weight=TVM_WEIGHT, tvm_iters=TVM_ITERS,
        gaussian_sigma=GAUSSIAN_SIGMA,
    )
    for dn in SOLO_DEFENSES:
        _record(f"{ATTACK_TAG}+{dn}", run_inference(atk_defs[dn]))

    ckpt.put(img_id, buckets)

start = time.time()
for i, fname in enumerate(tqdm(files, desc=f"{ATTACK_TAG}")):
    process_one(fname)
    if (i + 1) % 500 == 0:
        torch.cuda.empty_cache()
ckpt.flush()
print(f"\nDone in {(time.time()-start)/60:.1f} min "
      f"({len(ckpt.data)} images cached).")


# ## 11. Assemble results + COCO evaluation

# ============================================================
# Assemble results (including offline ensemble merging) + COCO eval
# ============================================================
all_results = pc.assemble_results(
    ckpt.data,
    defense_names=SOLO_DEFENSES,
    attack_tags=[ATTACK_TAG],
    ensembles=ENSEMBLES,
    nms_iou=ENSEMBLE_NMS_IOU,
)

cached_ids = sorted(ckpt.data.keys())
eval_stats = pc.evaluate_all_conditions(
    all_results, coco_gt, image_ids=cached_ids, output_dir=OUTPUT_DIR,
)

with open(os.path.join(OUTPUT_DIR, "eval_stats.pkl"), "wb") as f:
    pickle.dump({k: v.tolist() for k, v in eval_stats.items()}, f)


# ## 12. Summary + head-to-head comparison table

# ============================================================
# Summary JSON + head-to-head comparison table
# ============================================================
clean_ap = float(eval_stats["clean"][0])
atk_ap   = float(eval_stats[ATTACK_TAG][0])
attack_drop = clean_ap - atk_ap

all_defense_names = list(SOLO_DEFENSES) + list(ENSEMBLES.keys())

summary = {
    "model": YOLO_MODEL,
    "attack": ATTACK_TAG,
    "epsilon": EPSILON if ATTACK_TAG != "patch" else None,
    "num_images": len(cached_ids),
    "clean_mAP": clean_ap,
    "clean_AP50": float(eval_stats["clean"][1]),
    "attacked_mAP": atk_ap,
    "attack_damage": attack_drop,
    "defenses": {},
}
for dn in all_defense_names:
    tag = f"{ATTACK_TAG}+{dn}"
    if tag not in eval_stats:
        continue
    def_ap   = float(eval_stats[tag][0])
    def_ap50 = float(eval_stats[tag][1])
    rec      = def_ap - atk_ap
    rec_pct  = (100.0 * rec / attack_drop) if attack_drop > 0 else 0.0
    summary["defenses"][dn] = {
        "mAP": def_ap, "AP50": def_ap50,
        "recovery": rec, "recovery_pct": rec_pct,
        "kind": "ENSEMBLE" if dn in ENSEMBLES else "solo",
    }

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

pc.print_banner(f"ATTACKED vs RECOVERED — {ATTACK_TAG} (YOLO)", width=100)
print(f"  Clean baseline   : {clean_ap:.4f}")
print(f"  Attacked         : {atk_ap:.4f}")
print(f"  Attack damage    : {attack_drop:+.4f}\n")

rows = []
for dn, info in summary["defenses"].items():
    rows.append((dn, info["kind"], info["mAP"], info["recovery"], info["recovery_pct"]))
rows.sort(key=lambda r: r[3], reverse=True)

print("-" * 100)
print(f"  {'Rank':<5} {'Defense':<28} {'Kind':<10} {'Attacked':>9} {'Recovered':>10} {'Δ mAP':>9} {'Rec%':>7} {'Verdict':>10}")
print("-" * 100)
for rank, (name, kind, def_ap, rec, rec_pct) in enumerate(rows, 1):
    if rec > 0.005:
        verdict = "RECOVERS"
    elif rec > 0:
        verdict = "marginal"
    elif rec > -0.005:
        verdict = "neutral"
    else:
        verdict = "HURTS"
    print(f"  {rank:<5} {name:<28} {kind:<10} {atk_ap:>9.4f} {def_ap:>10.4f} {rec:>+9.4f} {rec_pct:>+6.1f}% {verdict:>10}")
print("-" * 100)

best  = rows[0]
worst = rows[-1]
best_pct = (100.0 * best[3] / attack_drop) if attack_drop > 0 else 0.0
print(f"\n  >> BEST  : {best[0]} ({best[1]})  mAP {atk_ap:.4f} -> {best[2]:.4f}  "
      f"(+{best[3]:.4f}, {best_pct:+.1f}%)")
print(f"  >> WORST : {worst[0]} ({worst[1]})  mAP {atk_ap:.4f} -> {worst[2]:.4f}  ({worst[3]:+.4f})")

solo_rec = [r[3] for r in rows if r[1] == "solo"]
ens_rec  = [r[3] for r in rows if r[1] == "ENSEMBLE"]
if solo_rec and ens_rec:
    print(f"\n  Solo     recovery  avg={np.mean(solo_rec):+.4f}  max={max(solo_rec):+.4f}  min={min(solo_rec):+.4f}")
    print(f"  Ensemble recovery  avg={np.mean(ens_rec):+.4f}  max={max(ens_rec):+.4f}  min={min(ens_rec):+.4f}")
    winner = "ENSEMBLING" if max(ens_rec) > max(solo_rec) else "SOLO"
    print(f"  >> Best-of-kind winner: {winner}")


# ## 13. Bar chart — Attacked vs Recovered

# ============================================================
# Horizontal bar chart: attacked vs recovered
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bar_rows = []
for dn in SOLO_DEFENSES:
    tag = f"{ATTACK_TAG}+{dn}"
    if tag in eval_stats:
        bar_rows.append((f"[solo] {dn}", float(eval_stats[tag][0]), "steelblue"))
for en in ENSEMBLES:
    tag = f"{ATTACK_TAG}+{en}"
    if tag in eval_stats:
        bar_rows.append((f"[ENS]  {en}", float(eval_stats[tag][0]), "darkorange"))
bar_rows.sort(key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(12, 0.45 * len(bar_rows) + 2))
y = np.arange(len(bar_rows))
mAPs = [r[1] for r in bar_rows]
colors = [r[2] for r in bar_rows]
ax.barh(y, mAPs, color=colors, alpha=0.85)
ax.axvline(x=atk_ap, color='red', linestyle='--', linewidth=2,
           label=f'Attacked ({ATTACK_TAG}): {atk_ap:.4f}')
ax.axvline(x=clean_ap, color='green', linestyle='--', linewidth=2,
           label=f'Clean: {clean_ap:.4f}')
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in bar_rows])
ax.set_xlabel('mAP')
ax.set_title(f'Attacked vs Recovered — {ATTACK_TAG} (YOLOv8x-worldv2)')
ax.legend(loc='lower right')
ax.grid(True, axis='x', alpha=0.3)
for yi, m in zip(y, mAPs):
    d = m - atk_ap
    col = 'darkgreen' if d > 0 else 'darkred'
    ax.text(m + 0.002, yi, f'{d:+.4f}', va='center', fontsize=8, color=col)
plt.tight_layout()
out_png = os.path.join(OUTPUT_DIR, f"attacked_vs_recovered_{ATTACK_TAG}.png")
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart saved -> {out_png}")


