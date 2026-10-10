import os
import torch

dataset = os.environ.get("DATASET", "1")
EMBEDDINGS_DIR = f"Dataset {dataset}/embeddings"

def get_item_ids(base_dir, images_dir=None, image_ext=".jpg"):
    """
    Uses img_embeddings_dinov2 as the source of truth for item IDs,
    then filters out any item that is missing:
      - its image file  (images/N.jpg)
      - any of the other three embedding files (cross-modality completeness)

    This guarantees load_modality_embeddings never hits a FileNotFoundError.

    Parameters
    ----------
    base_dir   : str  — root embeddings directory
    images_dir : str or None — path to the images folder.
                 Defaults to base_dir/../images/ relative to base_dir
    image_ext  : str  — image file extension to check for (default: ".jpg")
    """
    if images_dir is None:
        # embeddings/ sits next to images/ at the project root
        images_dir = os.path.join(os.path.dirname(base_dir), "images")
        if not os.path.isdir(images_dir):
            # fallback: base_dir/images/
            images_dir = os.path.join(base_dir, "images")

    embeddings_folder = os.path.join(base_dir, "img_embeddings_dinov2")
    all_item_ids = sorted(os.listdir(embeddings_folder))

    # All other modality folders that must also contain the .pt file
    other_modality_folders = [
        os.path.join(base_dir, "img_embeddings_fashionclip"),
        os.path.join(base_dir, "txt_embeddings_fashionclip"),
        os.path.join(base_dir, "txt_embeddings_qwen"),
    ]

    filtered = []
    skip_reasons = {}   # item_id -> reason string (for summary)

    for item_id in all_item_ids:
        stem = os.path.splitext(item_id)[0]
        # Embedding files may have an _aug suffix; strip it to get the image stem
        image_stem = stem[:-4] if stem.endswith("_aug") else stem

        # 1. Image file must exist
        image_path = os.path.join(images_dir, image_stem + image_ext)
        if not os.path.exists(image_path):
            skip_reasons[item_id] = f"missing image ({image_path})"
            continue

        # 2. Every other modality embedding must exist
        missing_mod = None
        for folder in other_modality_folders:
            if not os.path.exists(os.path.join(folder, item_id)):
                missing_mod = os.path.basename(folder)
                break

        if missing_mod:
            skip_reasons[item_id] = f"missing embedding in {missing_mod}"
            continue

        filtered.append(item_id)

    skipped = len(all_item_ids) - len(filtered)
    if skipped:
        print(f"[WARNING] Skipped {skipped} items due to missing files:")
        from collections import Counter
        reason_counts = Counter(r.split("(")[0].strip() for r in skip_reasons.values())
        for reason, count in reason_counts.items():
            print(f"  {count}x — {reason}")

    print(f"[INFO] {len(filtered)} / {len(all_item_ids)} items retained after completeness check.")
    return filtered


def load_modality_embeddings(base_dir, folder_name, item_ids):
    embeddings = []

    folder_path = os.path.join(base_dir, folder_name)
    nan_count  = 0
    zero_count = 0

    for item_id in item_ids:
        emb = torch.load(os.path.join(folder_path, item_id))

        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)

        if torch.isnan(emb).any():
            emb = torch.nan_to_num(emb, nan=0.0)
            nan_count += 1

        if emb.norm() < 1e-8:
            emb = torch.zeros_like(emb)
            zero_count += 1

        embeddings.append(emb)

    if nan_count or zero_count:
        print(f"  [WARNING] {folder_name}: {nan_count} NaN vectors, "
              f"{zero_count} zero-norm vectors — replaced with zeros.")

    return torch.stack(embeddings)


def load_all_embeddings(base_dir=EMBEDDINGS_DIR, images_dir=None, image_ext=".jpg"):
    item_ids = get_item_ids(base_dir, images_dir=images_dir, image_ext=image_ext)

    embeddings = {
        "item_ids": item_ids,
        "image_dino": load_modality_embeddings(
            base_dir, "img_embeddings_dinov2", item_ids
        ),
        "image_clip": load_modality_embeddings(
            base_dir, "img_embeddings_fashionclip", item_ids
        ),
        "text_clip": load_modality_embeddings(
            base_dir, "txt_embeddings_fashionclip", item_ids
        ),
        "text_qwen": load_modality_embeddings(
            base_dir, "txt_embeddings_qwen", item_ids
        ),
    }

    return embeddings