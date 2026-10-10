import os
import torch
from fusion.self_attention import MultiModalSelfAttentionFusion
from training._train_base import run_training
from load_embeddings import load_all_embeddings

dataset = os.environ.get("DATASET", "1")
EMBEDDINGS_DIR = f"Dataset {dataset}/embeddings"


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = dict(
        BATCH_SIZE           = 128,
        PHASE1_EPOCHS        = 5,
        PHASE2_EPOCHS        = 50,
        LR_PHASE1            = 3e-4,
        LR_PHASE2            = 1e-4,
        WARMUP_EPOCHS        = 2,
        TEMPERATURE          = 0.07,
        LAMBDA_UNIFORMITY    = 0.3,
        SIMILARITY_THRESHOLD = 0.6,
        PSEUDO_COS_THRESHOLD = 0.85,
        SOFT_LABEL_THRESHOLD = 0.3,
        SOFT_WEIGHT          = 0.5,
        CATEGORIES_PER_BATCH = 16,
        MAX_GRAD_NORM        = 1.0,
        DEVICE               = DEVICE,
        modality_keys        = ["image_clip", "image_dino", "text_clip", "text_qwen"],
    )

    emb        = load_all_embeddings(EMBEDDINGS_DIR)
    input_dims = {k: emb[k].shape[1] for k in cfg["modality_keys"]}

    model = MultiModalSelfAttentionFusion(
        input_dims=input_dims,
        d_model=1024,
        output_dim=1024,
        nhead=8,
        num_layers=2,
        dropout=0.1,
    ).to(DEVICE)

    MODELS_DIR = f"Dataset {dataset}/models"
    run_training(model, f"{MODELS_DIR}/selfattention_trained.pt", cfg)