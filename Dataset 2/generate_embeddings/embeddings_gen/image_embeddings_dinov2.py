import torch
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

image_dir = Path("../dataset2/images")
embedding_dir = Path("../dataset2/img_embeddings_dinov2")
embedding_dir.mkdir(parents=True, exist_ok=True)

model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_dinov2_embedding(image_path, save_dir):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output

    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    save_path = save_dir / f"{image_path.stem}.pt"
    torch.save(embedding, save_path)
    print(f"Saved: {save_path}")

image_extensions = {".jpg", ".jpeg", ".png"}
for image_path in image_dir.iterdir():
    if image_path.suffix.lower() in image_extensions:
        generate_dinov2_embedding(image_path, embedding_dir)

print("\n DINOv2 embeddings saved successfully!")
