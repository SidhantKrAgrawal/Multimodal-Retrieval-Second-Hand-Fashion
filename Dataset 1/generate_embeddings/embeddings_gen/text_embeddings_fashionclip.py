import torch
from pathlib import Path
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

captions_file = Path("captions_updated.txt")
text_embedding_dir = Path("../dataset1/txt_embeddings_fashionclip")
text_embedding_dir.mkdir(parents=True, exist_ok=True)

model_name = "patrickjohncyh/fashion-clip"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)

def generate_fashionclip_text_embedding(caption_text, key, save_dir):
    inputs = processor(text=caption_text, return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)
    save_path = save_dir / f"front_{key}_aug.pt"
    torch.save(text_embedding, save_path)
    print(f"Saved text embedding: {save_path}")

with open(captions_file, "r") as f:
    lines = f.readlines()

for line in lines:
    if ":" in line:
        key, caption = line.split(":", 1)
        key = key.strip()
        caption = caption.strip()
        generate_fashionclip_text_embedding(caption, key, text_embedding_dir)

print("\nFashionCLIP text embeddings saved successfully!")
