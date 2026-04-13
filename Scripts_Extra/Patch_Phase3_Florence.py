# # Adversarial Patch Phase 3 — Florence-2-Base
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
from transformers import AutoProcessor, AutoModelForCausalLM
from pycocotools.coco import COCO

import phase3_common as pc

warnings.filterwarnings("ignore")
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | GPUs visible: {torch.cuda.device_count()}")


# ## 2. GPU diagnostics

assert torch.cuda.is_available(), "CUDA required for Phase 3."
device = torch.device("cuda:0")
torch_dtype = torch.float16
props = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info(0)
print(f"Device: {device} -> {props.name}")
print(f"Memory: {free/1024**3:.2f} GB free / {total/1024**3:.2f} GB total")
print(f"Model dtype: {torch_dtype}")


# ## 3. Configuration

# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_DIR = "./Dataset/val2017"
ANN_FILE  = "./Dataset/annotations/instances_val2017.json"

NUM_IMAGES       = 3000
CHECKPOINT_EVERY = 1000

# Attack config
ATTACK_TAG = "patch"
EPSILON    = 0.03         # FGSM / PGD in normalized pixel space
PGD_ITERS  = 10
PGD_ALPHA  = EPSILON / 4.0

# Patch config
PATCH_SIZE      = pc.PATCH_SIZE        # 35
PATCH_OPT_ITERS = pc.PATCH_OPT_ITERS   # 100
PATCH_LR        = pc.PATCH_LR          # 0.02

# Defense hyperparams (matched to VariantZ)
JPEG_QUALITY     = 75
MEDIAN_KERNEL    = 3
TVM_WEIGHT       = 0.05
TVM_ITERS        = 200
GAUSSIAN_SIGMA   = 1.0
NMS_IOU_THRESHOLD = 0.5
ENSEMBLE_NMS_IOU  = 0.5

SOLO_DEFENSES = pc.SOLO_DEFENSES
ENSEMBLES     = pc.ENSEMBLES

OUTPUT_DIR      = "./results_phase3_florence_patch"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "detections.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

pc.print_banner(f"Phase 3 Florence-2 — {ATTACK_TAG}", width=70)
print(f"Images: {NUM_IMAGES} | Attack: {ATTACK_TAG} | eps={EPSILON}")
print(f"Solo defenses ({len(SOLO_DEFENSES)}): {SOLO_DEFENSES}")
print(f"Ensembles     ({len(ENSEMBLES)}):")
for n, m in ENSEMBLES.items():
    print(f"  {n:<30s} = {' + '.join(m)}")
print(f"Output: {OUTPUT_DIR}")


# ## 4. Load Florence-2 + COCO GT

# ============================================================
# Load Florence-2-Base + COCO GT
# ============================================================
model_name = "microsoft/Florence-2-base"
revision   = "refs/pr/26"

processor = AutoProcessor.from_pretrained(model_name, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, revision=revision,
    torch_dtype=torch_dtype, trust_remote_code=True,
).to(device).eval()

IMG_MEAN = torch.tensor(processor.image_processor.image_mean,
                        device=device, dtype=torch_dtype).view(1, 3, 1, 1)
IMG_STD  = torch.tensor(processor.image_processor.image_std,
                        device=device, dtype=torch_dtype).view(1, 3, 1, 1)

free, total = torch.cuda.mem_get_info(0)
print(f"Florence-2 loaded — {(total-free)/1024**2:.0f} MB VRAM used.")

coco_gt = COCO(ANN_FILE)
categories = coco_gt.loadCats(coco_gt.getCatIds())
category_mapping = {c["name"]: c["id"] for c in categories}

files = sorted(os.listdir(IMAGE_DIR))
if NUM_IMAGES is not None:
    files = files[:NUM_IMAGES]
evaluated_img_ids = sorted({int(os.path.splitext(f)[0]) for f in files})
print(f"COCO: {len(category_mapping)} categories | {len(files)} images selected.")


# ## 5. Label mapping + heuristic scores + inference

# ============================================================
# Florence-2 -> COCO Label Mapping + heuristic scores + NMS
# (Mirrors VariantZ cell 10 exactly — do not change, used by all notebooks.)
# ============================================================
FLORENCE_TO_COCO = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "baby": "person", "kid": "person", "player": "person",
    "pedestrian": "person", "human": "person", "skier": "person",
    "snowboarder": "person", "surfer": "person", "rider": "person",
    "automobile": "car", "van": "car", "sedan": "car", "suv": "car",
    "taxi": "car", "minivan": "car",
    "motor bike": "motorcycle", "motorbike": "motorcycle",
    "aeroplane": "airplane", "aircraft": "airplane", "jet": "airplane",
    "lorry": "truck", "pickup truck": "truck",
    "television": "tv", "tv set": "tv", "monitor": "tv", "screen": "tv",
    "television set": "tv",
    "mobile phone": "cell phone", "cellphone": "cell phone",
    "smartphone": "cell phone", "phone": "cell phone",
    "computer keyboard": "keyboard", "computer mouse": "mouse",
    "notebook computer": "laptop", "notebook": "laptop",
    "studio couch": "couch", "sofa": "couch", "settee": "couch",
    "kitchen & dining room table": "dining table", "table": "dining table",
    "desk": "dining table",
    "swivel chair": "chair", "armchair": "chair", "stool": "chair",
    "puppy": "dog", "kitten": "cat",
    "ski": "skis", "ski pole": "skis",
    "racket": "tennis racket",
    "ball": "sports ball", "football": "sports ball",
    "soccer ball": "sports ball", "baseball": "sports ball",
    "basketball": "sports ball", "tennis ball": "sports ball",
    "glove": "baseball glove",
    "houseplant": "potted plant", "plant": "potted plant", "flower pot": "potted plant",
    "flowerpot": "potted plant",
    "wine bottle": "bottle", "beer bottle": "bottle", "water bottle": "bottle",
    "drinking glass": "wine glass", "glass": "wine glass", "goblet": "wine glass",
    "pocketknife": "knife", "kitchen knife": "knife", "butter knife": "knife",
    "hair dryer": "hair drier", "hairdryer": "hair drier", "blow dryer": "hair drier",
    "wristwatch": "clock", "wall clock": "clock", "alarm clock": "clock",
    "bag": "handbag", "purse": "handbag", "wristlet": "handbag",
    "briefcase": "suitcase", "luggage": "suitcase", "travel bag": "suitcase",
    "backpack bag": "backpack",
    "traffic signal": "traffic light",
    "fire plug": "fire hydrant",
}

def _map_label(label):
    if label in category_mapping: return label
    m = FLORENCE_TO_COCO.get(label)
    if m and m in category_mapping: return m
    lo = label.lower()
    if lo in category_mapping: return lo
    m2 = FLORENCE_TO_COCO.get(lo)
    if m2 and m2 in category_mapping: return m2
    return None

def _compute_score(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    img_area = img_w * img_h
    box_area = w * h
    area_ratio = min(box_area / img_area, 0.5) if img_area > 0 else 0
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    icx, icy = img_w / 2, img_h / 2
    cd = np.sqrt(((cx - icx) / img_w) ** 2 + ((cy - icy) / img_h) ** 2)
    s = 0.6 + 0.2 * area_ratio + 0.15 * (1 - cd)
    return min(0.98, max(0.6, s))

def _box_iou_local(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    u = aa + ab - inter
    return inter / u if u > 0 else 0

def non_max_suppression(boxes, labels, scores, iou_thr=0.5):
    if not boxes: return [], [], []
    boxes = np.array(boxes)
    idxs = np.argsort(scores)[::-1]
    keep, kl, ks = [], [], []
    for i in idxs:
        sup = False
        for j in keep:
            if labels[i] == labels[j] and _box_iou_local(boxes[i], boxes[j]) > iou_thr:
                sup = True; break
        if not sup:
            keep.append(i); kl.append(labels[i]); ks.append(scores[i])
    return boxes[keep].tolist(), kl, ks

def run_inference(pil_img):
    img_w, img_h = pil_img.size
    with torch.no_grad():
        inputs = processor(text="<OD>", images=pil_img, return_tensors="pt")
        input_ids    = inputs.input_ids.to(device)
        pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype)
        gen_ids = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=512, num_beams=5, do_sample=False,
            repetition_penalty=1.8, length_penalty=1.0,
        )
        txt = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            txt, task="<OD>", image_size=(img_w, img_h)
        ) or {}
    od = parsed.get("<OD>", {})
    bboxes, labels = od.get("bboxes", []), od.get("labels", [])
    scores = [_compute_score(b, img_w, img_h) for b in bboxes]
    kb, kl, ks = non_max_suppression(bboxes, labels, scores, iou_thr=NMS_IOU_THRESHOLD)
    results = []
    for box, lab, sc in zip(kb, kl, ks):
        mp = _map_label(lab)
        if mp is None: continue
        cid = category_mapping[mp]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: continue
        results.append({"bbox": [x1, y1, w, h], "category_id": cid, "score": sc})
    return results

print("Inference pipeline ready (Florence-2 -> COCO).")


# ## 6. Patch Attack

# ============================================================
# Adversarial Patch (Florence-2) — 35x35 center-placed, 100 opt steps
# Optimizes patch pixels to maximize cross-entropy loss on the model's
# own beam-search prediction (untargeted). Applied at the Florence-2
# input resolution (normalized-pixel space), then decoded back to PIL.
# ============================================================
def patch_attack(pil_img, patch_size=PATCH_SIZE,
                 iters=PATCH_OPT_ITERS, lr=PATCH_LR):
    orig_size = pil_img.size
    inputs = processor(text="<OD>", images=pil_img, return_tensors="pt")
    input_ids    = inputs.input_ids.to(device)
    pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype)

    with torch.no_grad():
        target_ids = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=512, num_beams=5, do_sample=False,
            repetition_penalty=1.8, length_penalty=1.0,
        )
    target_ids = target_ids[:1, :].contiguous()
    if target_ids.size(1) > 512:
        target_ids = target_ids[:, :512].contiguous()

    H, W = pixel_values.shape[-2:]
    top, left = pc.center_patch_coords(H, W, patch_size)

    # Work in [0,1] pixel space for the patch (easier to constrain),
    # compose into the normalized tensor at each step.
    torch.manual_seed(0)
    patch01 = (torch.rand(3, patch_size, patch_size,
                          device=device, dtype=torch_dtype) * 0.5 + 0.25)
    patch01 = patch01.detach().clone()
    patch01.requires_grad_(True)
    optim = torch.optim.Adam([patch01], lr=lr)

    mean3 = IMG_MEAN.view(3, 1, 1)
    std3  = IMG_STD.view(3, 1, 1)

    for _ in range(iters):
        optim.zero_grad()
        patch_norm = (patch01.clamp(0.0, 1.0) - mean3) / std3
        composed = pixel_values.clone()
        composed[:, :, top:top + patch_size, left:left + patch_size] = patch_norm.unsqueeze(0)
        out = model(input_ids=input_ids, pixel_values=composed, labels=target_ids)
        # Maximize loss -> negate for minimization by optimizer
        (-out.loss).backward()
        optim.step()

    with torch.no_grad():
        patch_final = patch01.clamp(0.0, 1.0)
        patch_norm_final = (patch_final - mean3) / std3
        adv = pixel_values.clone()
        adv[:, :, top:top + patch_size, left:left + patch_size] = patch_norm_final.unsqueeze(0)

    adv_denorm = (adv.squeeze(0) * IMG_STD.squeeze(0) + IMG_MEAN.squeeze(0)).clamp(0.0, 1.0)
    adv_np = (adv_denorm.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    if adv_pil.size != orig_size:
        adv_pil = adv_pil.resize(orig_size, Image.BICUBIC)
    return adv_pil

print(f"Patch attack ready ({PATCH_SIZE}x{PATCH_SIZE}, {PATCH_OPT_ITERS} iters).")


# ## 7. Bind `attack_fn` for the shared main loop

attack_fn = patch_attack
print(f'attack_fn bound to {attack_fn.__name__}')


# ## 8. Sanity check

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


# ## 9. Main evaluation loop (checkpointed every 1000 images)

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

    _record("clean", run_inference(pil_img))
    clean_defs = pc.apply_all_defenses_gpu(
        pil_img, device,
        jpeg_quality=JPEG_QUALITY, median_kernel=MEDIAN_KERNEL,
        tvm_weight=TVM_WEIGHT, tvm_iters=TVM_ITERS,
        gaussian_sigma=GAUSSIAN_SIGMA,
    )
    for dn in SOLO_DEFENSES:
        _record(f"clean+{dn}", run_inference(clean_defs[dn]))

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
    if (i + 1) % 250 == 0:
        torch.cuda.empty_cache()
ckpt.flush()
print(f"\nDone in {(time.time()-start)/60:.1f} min "
      f"({len(ckpt.data)} images cached).")


# ## 10. Assemble results + COCO evaluation

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


# ## 11. Summary + head-to-head comparison table

# ============================================================
# Summary JSON + head-to-head comparison table
# ============================================================
clean_ap = float(eval_stats["clean"][0])
atk_ap   = float(eval_stats[ATTACK_TAG][0])
attack_drop = clean_ap - atk_ap

all_defense_names = list(SOLO_DEFENSES) + list(ENSEMBLES.keys())

summary = {
    "model": "microsoft/Florence-2-base",
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

pc.print_banner(f"ATTACKED vs RECOVERED — {ATTACK_TAG} (FLORENCE-2)", width=100)
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


# ## 12. Bar chart — Attacked vs Recovered

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
ax.set_title(f'Attacked vs Recovered — {ATTACK_TAG} (Florence-2-Base)')
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


