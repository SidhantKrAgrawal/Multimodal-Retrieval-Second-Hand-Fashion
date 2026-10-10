import json
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import CLIPTokenizer, CLIPModel

_here              = Path(__file__).parent
labels_dir         = (_here / "../../labels").resolve()
text_embedding_dir = (_here / "../../embeddings/txt_embeddings_fashionclip").resolve()
text_embedding_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# patrickjohncyh/fashion-clip produces NaN outputs with transformers >= 5.x due to position_ids changes
model_name = "openai/clip-vit-base-patch32"
tokenizer  = CLIPTokenizer.from_pretrained(model_name)
model      = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

BATCH_SIZE = 64

_EMPTY = {"", "n/a", "nan", "none", "other", "unknown"}

def _val(v) -> str:
    s = str(v).strip()
    return "" if s.lower() in _EMPTY else s


_COND_MAP = {"1": "very poor condition", "2": "poor condition", "3": "fair condition",
             "4": "good condition", "5": "like new"}

def build_caption(label: dict) -> str:
    brand     = _val(label.get("brand", ""))
    item_type = _val(label.get("type", ""))
    colors    = [_val(c) for c in label.get("colors", []) if _val(c)]
    category  = _val(label.get("category", ""))
    size      = _val(label.get("size", ""))
    season    = _val(label.get("season", ""))
    material  = _val(label.get("material", ""))
    pattern   = _val(label.get("pattern", ""))
    cut       = [_val(c) for c in label.get("cut", []) if _val(c)]
    trend     = _val(label.get("trend", ""))
    condition = _val(str(label.get("condition", "")))
    comment   = _val(label.get("comment", ""))

    cond_str = _COND_MAP.get(condition, "")

    attrs = []
    if brand and brand.lower() != "not in the list": attrs.append(brand)
    if item_type: attrs.append(item_type)
    if colors:    attrs.append(", ".join(colors))
    if category:  attrs.append(category)
    if size:      attrs.append(f"size {size}")
    if season:    attrs.append(f"{season} wear")
    if material:  attrs.append(material)
    if pattern and pattern.lower() != "none":  attrs.append(pattern)
    if cut:       attrs.append(" ".join(cut))
    if trend and trend.lower() != "none":      attrs.append(trend)
    if cond_str:  attrs.append(cond_str)
    if comment:   attrs.append(comment)

    caption = ". ".join(attrs) if attrs else "clothing item"
    return caption[:500]


def generate_batch_embeddings(text_batch, keys, save_dir):
    inputs = tokenizer(
        text_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)

    with torch.no_grad():
        text_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        text_features = model.text_projection(text_out.pooler_output)

    # L2 normalise
    text_features = F.normalize(text_features, p=2, dim=-1)
    text_features = text_features.cpu()

    for emb, key in zip(text_features, keys):
        save_path = save_dir / f"{key}.pt"
        torch.save(emb, save_path)


if not labels_dir.exists():
    print(f"Labels directory not found at {labels_dir}")
    print("Run make1.py first to generate the JSON label files.")
else:
    json_files = sorted(
        [f for f in os.listdir(labels_dir) if f.endswith(".json")],
        key=lambda f: int(os.path.splitext(f)[0])
    )
    print(f"Found {len(json_files)} label files in '{labels_dir}'")

    print("\nCaption preview (first 3):")
    for fname in json_files[:3]:
        with open(labels_dir / fname) as f:
            label = json.load(f)
        key = os.path.splitext(fname)[0]
        print(f"  [{key}] {build_caption(label)}\n")

    texts, keys = [], []
    empty_count = 0

    for fname in json_files:
        key = os.path.splitext(fname)[0]
        with open(labels_dir / fname, encoding="utf-8") as f:
            label = json.load(f)

        caption = build_caption(label)
        if caption == "clothing item":
            empty_count += 1

        texts.append(caption)
        keys.append(key)

        if len(texts) == BATCH_SIZE:
            generate_batch_embeddings(texts, keys, text_embedding_dir)
            texts, keys = [], []

    if texts:
        generate_batch_embeddings(texts, keys, text_embedding_dir)

    if empty_count:
        print(f"\n[WARNING] {empty_count} items had no usable text — used 'clothing item' fallback.")

    print(f"\n✅ FashionCLIP text embeddings saved to {text_embedding_dir}")