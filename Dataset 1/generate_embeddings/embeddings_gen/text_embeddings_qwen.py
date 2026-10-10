import torch
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModel

captions_file = Path("captions_updated.txt")
text_embedding_dir = Path("../dataset1/txt_embeddings_qwen")
text_embedding_dir.mkdir(parents=True, exist_ok=True)

model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_qwen_text_embedding(caption_text, key, save_dir):
    inputs = tokenizer(caption_text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    # mean-pool token embeddings into a single fixed-size vector
    embedding = outputs.last_hidden_state.mean(dim=1)
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    save_path = save_dir / f"front_{key}_aug.pt"
    torch.save(embedding, save_path)
    print(f"Saved text embedding: {save_path}")

with open(captions_file, "r") as f:
    lines = f.readlines()

for line in lines:
    if ":" in line:
        key, caption = line.split(":", 1)
        key = key.strip()
        caption = caption.strip()
        generate_qwen_text_embedding(caption, key, text_embedding_dir)

print("\nQwen 1.5B instruct text embeddings saved successfully!")
