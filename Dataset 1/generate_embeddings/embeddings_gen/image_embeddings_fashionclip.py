import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

image_dir = Path("../dataset1/augmented_images")
embedding_dir = Path("../dataset1/image_embeddings")
embedding_dir.mkdir(parents=True, exist_ok=True)

model_name = "patrickjohncyh/fashion-clip"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)

def generate_fashionclip_embedding(image_path, save_dir):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    save_path = save_dir / f"{image_path.stem}.pt"
    torch.save(embedding, save_path)
    print(f"Saved: {save_path}")

image_extensions = {".jpg", ".jpeg", ".png"}
for image_path in image_dir.iterdir():
    if image_path.suffix.lower() in image_extensions:
        generate_fashionclip_embedding(image_path, embedding_dir)

