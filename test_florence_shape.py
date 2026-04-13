import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

device = "cuda:0"
model_name = "microsoft/Florence-2-base"
revision = "refs/pr/26"

processor = AutoProcessor.from_pretrained(model_name, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, trust_remote_code=True).to(device)

pil_img = Image.new("RGB", (224, 224), (255,255,255))
inputs = processor(text="<OCR>", images=pil_img, return_tensors="pt")
input_ids = inputs.input_ids.to(device)
pixel_values = inputs.pixel_values.to(device)

target_ids = model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=10)
print(f"input_ids shape: {input_ids.shape}")
print(f"target_ids shape: {target_ids.shape}")

try:
    out = model(input_ids=input_ids, pixel_values=pixel_values, labels=target_ids)
    print("Forward pass succeeded!")
except Exception as e:
    print("Forward pass FAILED:", e)

try:
    # Proper teacher forcing
    out2 = model(input_ids=target_ids, pixel_values=pixel_values, labels=target_ids)
    print("Teacher forcing forward pass succeeded!")
except Exception as e:
    print("Teacher forcing FAILED:", e)

