import sys
import torch
import torch.nn.functional as F
from load_embeddings import load_all_embeddings
import numpy as np


def concat_fusion(embeddings, modalities):
    normalized = [
        F.normalize(embeddings[m], dim=1)
        for m in modalities
    ]
    fused = torch.cat(normalized, dim=1)
    return F.normalize(fused, dim=1)


def concat_fusion_weighted(embeddings, modalities, weights):
    scaled = [
        F.normalize(embeddings[m], dim=1) * weights.get(m, 1.0)
        for m in modalities
    ]
    fused = torch.cat(scaled, dim=1)
    return F.normalize(fused, dim=1)


MODALITIES = ["image_dino", "image_clip", "text_clip", "text_qwen"]

WEIGHTS_TEXT_HEAVY = {
    "image_dino": 0.5,
    "image_clip": 0.5,
    "text_clip":  1.5,
    "text_qwen":  1.5,
}

WEIGHTS_IMAGE_HEAVY = {
    "image_dino": 1.5,
    "image_clip": 1.5,
    "text_clip":  0.5,
    "text_qwen":  0.5,
}

ALL_WEIGHT_PRESETS = {
    "text_heavy":  WEIGHTS_TEXT_HEAVY,
    "image_heavy": WEIGHTS_IMAGE_HEAVY,
}


def fashionclip_joint_fusion(embeddings):
    img = F.normalize(embeddings["image_clip"], dim=1)
    txt = F.normalize(embeddings["text_clip"], dim=1)
    return F.normalize(img + txt, dim=1)


if __name__ == "__main__":
    import os
    dataset        = os.environ.get("DATASET", "1")
    EMBEDDINGS_DIR       = f"Dataset {dataset}/embeddings"
    FUSED_EMBEDDINGS_DIR = f"Dataset {dataset}/fused_embeddings"
    os.makedirs(FUSED_EMBEDDINGS_DIR, exist_ok=True)

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("Loading embeddings...")
    emb = load_all_embeddings(EMBEDDINGS_DIR)

    if mode in ("all", "equal"):
        print("\n[1/2] Equal-weight concat...")
        fused    = concat_fusion(emb, MODALITIES)
        fused_np = fused.numpy().astype(np.float32)
        print(f"  Shape: {fused_np.shape}")
        path = f"{FUSED_EMBEDDINGS_DIR}/fused_item_embeddings_concat.npy"
        np.save(path, fused_np)
        print(f"  Saved → {path}")

    if mode in ("all", "weighted"):
        print("\n[2/3] Weighted concat variants...")
        for preset_name, weights in ALL_WEIGHT_PRESETS.items():
            print(f"\n  Preset: {preset_name}")
            print(f"  Weights: {weights}")
            fused    = concat_fusion_weighted(emb, MODALITIES, weights)
            fused_np = fused.numpy().astype(np.float32)
            print(f"  Shape: {fused_np.shape}")
            path = f"{FUSED_EMBEDDINGS_DIR}/fused_item_embeddings_concat_{preset_name}.npy"
            np.save(path, fused_np)
            print(f"  Saved → {path}")

    if mode in ("all", "joint"):
        print("\n[3/3] FashionCLIP joint embedding (image_clip + text_clip)...")
        fused    = fashionclip_joint_fusion(emb)
        fused_np = fused.numpy().astype(np.float32)
        print(f"  Shape: {fused_np.shape}")
        path = f"{FUSED_EMBEDDINGS_DIR}/fused_item_embeddings_fashionclip_joint.npy"
        np.save(path, fused_np)
        print(f"  Saved → {path}")

    print("\nDone.")