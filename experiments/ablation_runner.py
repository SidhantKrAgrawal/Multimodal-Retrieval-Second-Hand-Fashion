import os
import sys
import numpy as np
import torch

dataset              = os.environ.get("DATASET", "1")
EMBEDDINGS_DIR       = f"Dataset {dataset}/embeddings"
LABELS_DIR           = f"Dataset {dataset}/labels"
FUSED_EMBEDDINGS_DIR = f"Dataset {dataset}/fused_embeddings"
RESULTS_DIR          = f"Dataset {dataset}/results"
ABLATION_MODELS_DIR  = f"Dataset {dataset}/ablation_models"

from load_embeddings import load_all_embeddings
from fusion.self_attention import MultiModalSelfAttentionFusion
from training._train_base import run_training
from retrieval.hnsw_search import HNSWSearch
from evaluation.load_labels import load_all_labels
from evaluation.align_labels import align_labels_with_embeddings
from evaluation.build_ground_truth import build_ground_truth
from evaluation.evaluate_i2i import evaluate_i2i
from evaluation.evaluate_profile import evaluate_profile

EVAL_KS   = (5, 10, 20)
PROFILE_K = 10

BASE_CFG = dict(
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
    USE_WARMSTART        = True,
    modality_keys        = ["image_clip", "image_dino", "text_clip", "text_qwen"],
)

ABLATIONS = {
    "full":             {},
    "no_uniformity":    {"LAMBDA_UNIFORMITY": 0.0},
    "no_warmstart":     {"USE_WARMSTART": False},
    "hard_labels_only": {"PSEUDO_COS_THRESHOLD": 2.0},  # cos sim ≤ 1, so pseudo-labels never fire
    "phase1_only":      {"PHASE2_EPOCHS": 0},
}

HNSW_EF_CONSTRUCTION = 200
HNSW_M               = 16
HNSW_EF_SEARCH       = 50

def build_model(input_dims: dict, device: str) -> MultiModalSelfAttentionFusion:
    return MultiModalSelfAttentionFusion(
        input_dims=input_dims,
        d_model=1024,
        output_dim=1024,
        nhead=8,
        num_layers=2,
        dropout=0.1,
    ).to(device)


def generate_embeddings(model, emb: dict, modality_keys: list, device: str) -> np.ndarray:
    model.eval()
    inputs = {k: emb[k].to(device) for k in modality_keys}
    with torch.no_grad():
        fused = model(inputs)
    return fused.cpu().numpy().astype(np.float32)


def build_hnsw(fused_np: np.ndarray) -> HNSWSearch:
    idx = HNSWSearch(dim=fused_np.shape[1])
    idx.build(fused_np,
              ef_construction=HNSW_EF_CONSTRUCTION,
              M=HNSW_M,
              ef_search=HNSW_EF_SEARCH)
    return idx


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset}")

    os.makedirs(ABLATION_MODELS_DIR, exist_ok=True)
    os.makedirs(FUSED_EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nLoading embeddings and labels...")
    emb            = load_all_embeddings(EMBEDDINGS_DIR)
    labels_dict    = load_all_labels(LABELS_DIR)
    aligned_labels = align_labels_with_embeddings(emb["item_ids"], labels_dict)
    ground_truth   = build_ground_truth(aligned_labels)

    modality_keys = BASE_CFG["modality_keys"]
    input_dims    = {k: emb[k].shape[1] for k in modality_keys}
    print(f"  Catalogue size : {len(emb['item_ids'])} items")
    print(f"  Modality dims  : {input_dims}")

    all_results = {}

    for name, overrides in ABLATIONS.items():
        print(f"\n{'='*60}")
        print(f"ABLATION: {name.upper()}")
        if overrides:
            print(f"  Overrides : {overrides}")
        else:
            print(f"  Full model — no overrides (baseline)")
        print(f"{'='*60}")

        cfg = {**BASE_CFG, **overrides, "DEVICE": DEVICE}

        model_path = os.path.join(ABLATION_MODELS_DIR, f"selfattention_{name}.pt")
        emb_path   = os.path.join(FUSED_EMBEDDINGS_DIR, f"ablation_selfattention_{name}.npy")

        model = build_model(input_dims, DEVICE)
        run_training(model, model_path, cfg)

        fused_np = generate_embeddings(model, emb, modality_keys, DEVICE)
        np.save(emb_path, fused_np)
        print(f"  Embeddings saved → {emb_path}  shape={fused_np.shape}")

        print("  Building HNSW index...")
        hnsw = build_hnsw(fused_np)

        print("  Running GT evaluation...")
        gt_results = evaluate_i2i(
            embeddings=fused_np,
            hnsw=hnsw,
            ground_truth=ground_truth,
            ks=EVAL_KS,
        )

        print("  GT results:")
        for metric, vals in gt_results.items():
            line = "  ".join(f"@{k}: {v:.4f}" for k, v in vals.items())
            print(f"    {metric:<12} {line}")

        print(f"  Running profile evaluation (k={PROFILE_K})...")
        profile = evaluate_profile(
            embeddings=fused_np,
            search_method=hnsw,
            aligned_labels=aligned_labels,
            k=PROFILE_K,
        )
        ild = profile["ild"]
        print(f"    ILD@{PROFILE_K}        {ild:.4f}")

        all_results[name] = {"gt": gt_results, "ild": ild}

    print(f"\n{'='*60}")
    print(f"ABLATION STUDY — COMPARISON TABLE  (Dataset {dataset})")
    print(f"{'='*60}")

    col_w = 20
    header = (
        f"{'Ablation':<{col_w}}"
        f"{'P@5':>7} {'P@10':>7} {'P@20':>7}"
        f"{'N@5':>8} {'N@10':>8} {'N@20':>8}"
        f"{'R@10':>7}"
        f"{'ILD':>7}"
    )
    print(header)
    print("-" * len(header))

    full_p10 = all_results["full"]["gt"]["precision"][10] if "full" in all_results else None

    for name in ABLATIONS:
        if name not in all_results:
            continue

        gt  = all_results[name]["gt"]
        ild = all_results[name]["ild"]

        p5,  p10,  p20  = gt["precision"][5],  gt["precision"][10],  gt["precision"][20]
        n5,  n10,  n20  = gt["ndcg"][5],       gt["ndcg"][10],       gt["ndcg"][20]
        r10             = gt["recall"][10]

        if name == "full":
            tag = " ← full"
        elif full_p10 is not None:
            delta = p10 - full_p10
            tag   = f" ({delta:+.4f})"
        else:
            tag = ""

        print(
            f"{name:<{col_w}}"
            f"{p5:>7.4f} {p10:>7.4f} {p20:>7.4f}"
            f"{n5:>8.4f} {n10:>8.4f} {n20:>8.4f}"
            f"{r10:>7.4f}"
            f"{ild:>7.3f}"
            f"{tag}"
        )

    print(f"\nNote: delta shown as (ablation P@10 − full P@10).")
    print(f"Negative delta = removing this component hurts performance.")
    print(f"\nRe-run with DATASET=2 to validate findings on the noisy dataset.")


if __name__ == "__main__":
    txt_path = f"Dataset {dataset}/results_ablation_runner.txt"
    os.makedirs(f"Dataset {dataset}", exist_ok=True)

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
        def flush(self):
            for s in self._streams:
                s.flush()

    with open(txt_path, "w") as log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__
    print(f"Output saved to {txt_path}")
