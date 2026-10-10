import os
import numpy as np

dataset              = os.environ.get("DATASET", "1")
EMBEDDINGS_DIR       = f"Dataset {dataset}/embeddings"
LABELS_DIR           = f"Dataset {dataset}/labels"
IMAGES_DIR           = f"Dataset {dataset}/images"
FUSED_EMBEDDINGS_DIR = f"Dataset {dataset}/fused_embeddings"

from load_embeddings import load_all_embeddings
from retrieval.hnsw_search import HNSWSearch
from retrieval.late_fusion import weighted_score_fusion
from visualization.pdf_results import create_retrieval_pdf
from evaluation.load_labels import load_all_labels
from evaluation.align_labels import align_labels_with_embeddings


N_QUERIES       = 10
RANDOM_SEED     = 42   # set to None for a fresh draw each run
QUERY_INDEX_MIN = 1
QUERY_INDEX_MAX = 4000

EARLY_METHODS = [
    "selfattention",
    "coattention",
    "crossattention",
    "concat",
    "concat_text_heavy",
    "concat_image_heavy",
]

EMBEDDING_FILES = {
    "selfattention":       "fused_item_embeddings_selfattention.npy",
    "coattention":         "fused_item_embeddings_coattention.npy",
    "crossattention":      "fused_item_embeddings_crossattention.npy",
    "concat":              "fused_item_embeddings_concat.npy",
    "concat_text_heavy":   "fused_item_embeddings_concat_text_heavy.npy",
    "concat_image_heavy":  "fused_item_embeddings_concat_image_heavy.npy",
}

RUN_LATE_FUSION        = True
LATE_FUSION_MODALITIES = ["image_clip", "image_dino", "text_clip", "text_qwen"]

LATE_FUSION_WEIGHTS = {
    "image_clip": 0.8,
    "image_dino": 0.8,
    "text_clip":  0.8,
    "text_qwen":  0.8,
}

LATE_FUSION_ALPHA    = 5.0
LATE_FUSION_SEARCH_K = 60

HNSW_EF_CONSTRUCTION = 200
HNSW_M               = 16
HNSW_EF_SEARCH       = 50

RESULTS_DIR      = f"Dataset {dataset}/results"
OUTPUT_ROOT      = "qualitative_results"
OUTPUT_ROOT_FULL = os.path.join(RESULTS_DIR, OUTPUT_ROOT)


def build_hnsw(fused_np: np.ndarray) -> HNSWSearch:
    idx = HNSWSearch(dim=fused_np.shape[1])
    idx.build(fused_np,
              ef_construction=HNSW_EF_CONSTRUCTION,
              M=HNSW_M,
              ef_search=HNSW_EF_SEARCH)
    return idx


def retrieve_early(search_method, fused_np, query_idx, k=10):
    query = fused_np[query_idx:query_idx + 1]
    labels, distances = search_method.search(query, k=k + 1)
    labels    = np.asarray(labels)
    distances = np.asarray(distances)
    if labels.ndim > 1:
        labels, distances = labels[0], distances[0]
    mask      = labels != query_idx
    return labels[mask][:k], distances[mask][:k]


def retrieve_late_fusion(raw_embeddings, per_mod_indexes, query_idx, k=10):
    results_buffer = {}
    for m, idx in per_mod_indexes.items():
        q = raw_embeddings[m][query_idx:query_idx + 1]
        labels, distances = idx.search(q, k=LATE_FUSION_SEARCH_K + 1)
        labels    = np.asarray(labels[0])
        distances = np.asarray(distances[0])
        mask = labels != query_idx
        results_buffer[m] = (labels[mask], distances[mask])

    fused = weighted_score_fusion(
        results_buffer, LATE_FUSION_WEIGHTS, alpha=LATE_FUSION_ALPHA)[:k]

    item_indices = [i for i, _ in fused]
    scores       = [s for _, s in fused]
    return item_indices, scores


def save_pdf(output_path, query_item_id, retrieved_indices,
             retrieved_item_ids, distances, aligned_labels, query_idx):
    full_dir = os.path.join(RESULTS_DIR, os.path.dirname(output_path))
    os.makedirs(full_dir, exist_ok=True)
    create_retrieval_pdf(
        output_path=output_path,
        query_item_id=query_item_id,
        retrieved_item_ids=retrieved_item_ids,
        images_dir=IMAGES_DIR,
        distances=distances,
        retrieved_metadata=[aligned_labels[i] for i in retrieved_indices],
        query_metadata=aligned_labels[query_idx],
        result_dir=RESULTS_DIR,
    )


def main():
    rng          = np.random.default_rng(RANDOM_SEED)
    query_indices = rng.integers(
        QUERY_INDEX_MIN, QUERY_INDEX_MAX + 1,
        size=N_QUERIES, endpoint=True
    ).tolist()
    # Remove duplicates while preserving order
    seen = set()
    query_indices = [q for q in query_indices
                     if not (q in seen or seen.add(q))]
    print(f"Sampled {len(query_indices)} unique query indices: {query_indices}")

    print("\nLoading embeddings and labels...")
    emb            = load_all_embeddings(EMBEDDINGS_DIR)
    labels_dict    = load_all_labels(LABELS_DIR)
    aligned_labels = align_labels_with_embeddings(emb["item_ids"], labels_dict)

    n_items = len(emb["item_ids"])
    print(f"  Catalogue size: {n_items} items")

    query_indices = [q for q in query_indices if q < n_items]
    print(f"  After clamping to [0, {n_items-1}]: {len(query_indices)} queries remain")

    print("\nBuilding HNSW indexes for early fusion models...")
    early_indexes = {}   # method -> (fused_np, HNSWSearch)

    for method in EARLY_METHODS:
        emb_file = os.path.join(FUSED_EMBEDDINGS_DIR, EMBEDDING_FILES[method])
        if not os.path.exists(emb_file):
            print(f"  [SKIP] {method}: embedding file not found ({emb_file})")
            continue

        fused_np = np.load(emb_file)
        print(f"  {method:<25} dim={fused_np.shape[1]}  building index...")
        search_method = build_hnsw(fused_np)
        early_indexes[method] = (fused_np, search_method)

    per_mod_indexes = None
    raw_embeddings  = None

    if RUN_LATE_FUSION:
        print("\nBuilding HNSW indexes for late fusion modalities...")
        raw_embeddings = {
            m: emb[m].numpy().astype(np.float32)
            for m in LATE_FUSION_MODALITIES
        }
        per_mod_indexes = {}
        for m, arr in raw_embeddings.items():
            idx = HNSWSearch(dim=arr.shape[1])
            idx.build(arr,
                      ef_construction=HNSW_EF_CONSTRUCTION,
                      M=HNSW_M,
                      ef_search=HNSW_EF_SEARCH)
            per_mod_indexes[m] = idx
            print(f"  {m:<20} dim={arr.shape[1]}")

    os.makedirs(OUTPUT_ROOT_FULL, exist_ok=True)

    print(f"\nGenerating PDFs → {OUTPUT_ROOT}/")
    print("=" * 60)

    for q_num, query_idx in enumerate(query_indices, start=1):
        query_item_id = emb["item_ids"][query_idx]
        folder_name   = f"Query_{query_idx}"
        query_dir      = os.path.join(OUTPUT_ROOT, folder_name)
        query_dir_full = os.path.join(OUTPUT_ROOT_FULL, folder_name)
        os.makedirs(query_dir_full, exist_ok=True)

        print(f"\n[{q_num}/{len(query_indices)}]  idx={query_idx}  "
              f"item={query_item_id}  → {folder_name}/")

        for method, (fused_np, search_method) in early_indexes.items():
            labels, distances = retrieve_early(search_method, fused_np, query_idx)
            label_ints = [int(i) for i in labels]

            pdf_path = os.path.join(query_dir, f"{method}.pdf")
            save_pdf(
                output_path=pdf_path,
                query_item_id=query_item_id,
                retrieved_indices=label_ints,
                retrieved_item_ids=[emb["item_ids"][i] for i in label_ints],
                distances=distances,
                aligned_labels=aligned_labels,
                query_idx=query_idx,
            )
            print(f"  ✓ {method}.pdf")

        if RUN_LATE_FUSION and per_mod_indexes is not None:
            item_indices, scores = retrieve_late_fusion(
                raw_embeddings, per_mod_indexes, query_idx)

            pdf_path = os.path.join(query_dir, "late_fusion.pdf")
            save_pdf(
                output_path=pdf_path,
                query_item_id=query_item_id,
                retrieved_indices=item_indices,
                retrieved_item_ids=[emb["item_ids"][i] for i in item_indices],
                distances=scores,
                aligned_labels=aligned_labels,
                query_idx=query_idx,
            )
            print(f"  ✓ late_fusion.pdf")

    print(f"\n{'='*60}")
    print(f"Done. Results written to: {OUTPUT_ROOT}/")
    print(f"  {len(query_indices)} query folders, each containing "
          f"{len(early_indexes) + (1 if RUN_LATE_FUSION else 0)} PDFs.")


if __name__ == "__main__":
    main()