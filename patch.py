import re

with open("FGSM_Phase3_Florence2_Hybrid.py", "r") as f:
    code = f.read()

# Fix docstring
code = re.sub(
    r'"""\nFGSM Phase 2.*?All from the "Feature Squeezing" family \(Xu et al\. NDSS 2018\)\.\n"""',
    '"""\nFGSM Phase 3 — Florence-2 Hybrid Novel Evaluation\n"""',
    code, flags=re.DOTALL
)

# Add skimage import
if 'from skimage.restoration import denoise_tv_chambolle' not in code:
    code = code.replace('from torchvision import transforms', 'from torchvision import transforms\nfrom skimage.restoration import denoise_tv_chambolle')

# Update constants
code = code.replace('NUM_IMAGES = 1000\n\n# FGSM epsilon values\nEPSILONS = [0.003, 0.01, 0.03]', 'NUM_IMAGES = 1000\n\n# FGSM epsilon values\nEPSILONS = [0.03]')
code = code.replace('OUTPUT_DIR = "./results_phase2_variantA"', 'OUTPUT_DIR = "./results_phase3_florence_hybrid"')

# Replace Defenses
defenses_block = """
# ============================================================
# 6. Defense Functions — Phase 3 Hybrid
# ============================================================

def _jpeg(pil_img, quality=75):
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def _median(pil_img, kernel=3):
    return pil_img.filter(ImageFilter.MedianFilter(size=kernel))

def _tvm(pil_img, weight=0.05):
    arr = np.array(pil_img).astype(np.float64) / 255.0
    denoised = denoise_tv_chambolle(arr, weight=weight, channel_axis=-1)
    return Image.fromarray((np.clip(denoised, 0, 1) * 255).astype(np.uint8))

def _random_cutout(pil_img, cutout_prob=0.15, block_size=16):
    arr = np.array(pil_img)
    h, w, c = arr.shape
    num_blocks_y = h // block_size
    num_blocks_x = w // block_size
    
    mask = np.random.rand(num_blocks_y, num_blocks_x) > cutout_prob
    mask_upsampled = mask.repeat(block_size, axis=0).repeat(block_size, axis=1)
    
    full_mask = np.ones((h, w), dtype=bool)
    full_mask[:mask_upsampled.shape[0], :mask_upsampled.shape[1]] = mask_upsampled
    
    arr[~full_mask] = 0
    return Image.fromarray(arr)

DEFENSES = {
    "jpeg_median_tvm_cutout": lambda img: _random_cutout(_tvm(_median(_jpeg(img)))),
    "jpeg_median_cutout": lambda img: _random_cutout(_median(_jpeg(img))),
    "jpeg_tvm_cutout": lambda img: _random_cutout(_tvm(_jpeg(img))),
    "median_tvm_cutout": lambda img: _random_cutout(_tvm(_median(img))),
    "jpeg_median": lambda img: _median(_jpeg(img)),
    "tvm_cutout": lambda img: _random_cutout(_tvm(img))
}
"""
code = re.sub(r'# ============================================================\n# 6\. Defense Functions — Variant A.*?print\(f"Defenses: \{list\(DEFENSES\.keys\(\)\)\}"\)', defenses_block.strip() + '\n\nprint(f"Defenses: {list(DEFENSES.keys())}")', code, flags=re.DOTALL)

with open("FGSM_Phase3_Florence2_Hybrid.py", "w") as f:
    f.write(code)

print("Patched completely!")
