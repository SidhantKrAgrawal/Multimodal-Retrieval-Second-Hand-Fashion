import os
import sys
import numpy as np

dataset              = os.environ.get("DATASET", "1")
EMBEDDINGS_DIR       = f"Dataset {dataset}/embeddings"
LABELS_DIR           = f"Dataset {dataset}/labels"
IMAGES_DIR           = f"Dataset {dataset}/images"
FUSED_EMBEDDINGS_DIR = f"Dataset {dataset}/fused_embeddings"
RESULTS_DIR          = f"Dataset {dataset}/results"

from load_embeddings import load_all_embeddings
from fusion.concat import fashionclip_joint_fusion
from retrieval.hnsw_search import HNSWSearch
from visualization.pdf_results import create_retrieval_pdf
from evaluation.load_labels import load_all_labels
from evaluation.align_labels import align_labels_with_embeddings
from evaluation.build_ground_truth import build_ground_truth
from evaluation.evaluate_i2i import evaluate_i2i
from evaluation.evaluate_profile import (
    evaluate_profile, print_profile_report,
)
from evaluation.statistical_tests import (
    collect_query_scores,
    run_significance_tests,
    print_significance_report,
    print_significance_summary_table,
    print_effect_size_summary,
)
from evaluation.gt_sensitivity import run_sensitivity_analysis

QUERY_INDEX = 999

EVAL_KS    = (5, 10, 20)
PROFILE_K  = 10

COMPARISON_METHODS = {
    "concat":           "fused_item_embeddings_concat.npy",
    "concat_text_heavy":"fused_item_embeddings_concat_text_heavy.npy",
    "selfattention":    "fused_item_embeddings_selfattention.npy",
}

RUN_SIGNIFICANCE_TESTS = True
SIGNIFICANCE_ALPHA     = 0.05

RUN_SENSITIVITY        = True
SENSITIVITY_METRIC     = "precision"
SENSITIVITY_K          = 10
SENSITIVITY_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
SENSITIVITY_N_PERTURB  = 20

HNSW_EF_CONSTRUCTION = 200
HNSW_M               = 16
HNSW_EF_SEARCH       = 50

def main():
    print("Loading embeddings and labels...")
    emb            = load_all_embeddings(EMBEDDINGS_DIR)
    labels_dict    = load_all_labels(LABELS_DIR)
    aligned_labels = align_labels_with_embeddings(emb["item_ids"], labels_dict)
    gt             = build_ground_truth(aligned_labels)
    print(f"  Catalogue size : {len(emb['item_ids'])} items")

    print("\nGenerating FashionCLIP joint embedding (image_clip + text_clip)...")
    fused_t  = fashionclip_joint_fusion(emb)
    fused_np = fused_t.numpy().astype(np.float32)
    print(f"  Shape : {fused_np.shape}  (512-d, FashionCLIP joint space)")

    os.makedirs(FUSED_EMBEDDINGS_DIR, exist_ok=True)
    emb_path = f"{FUSED_EMBEDDINGS_DIR}/fused_item_embeddings_fashionclip_joint.npy"
    np.save(emb_path, fused_np)
    print(f"  Saved → {emb_path}")

    print("\nBuilding HNSW index...")
    hnsw = HNSWSearch(dim=fused_np.shape[1])
    hnsw.build(fused_np,
               ef_construction=HNSW_EF_CONSTRUCTION,
               M=HNSW_M,
               ef_search=HNSW_EF_SEARCH)

    print(f"\nExample query (idx={QUERY_INDEX}):")
    query = fused_np[QUERY_INDEX:QUERY_INDEX + 1]
    labels, distances = hnsw.search(query, k=11)
    labels    = np.asarray(labels[0])
    distances = np.asarray(distances[0])
    mask      = labels != QUERY_INDEX
    labels    = labels[mask][:10]
    distances = distances[mask][:10]

    print(f"  Query : {emb['item_ids'][QUERY_INDEX]}")
    print("  Retrieved:")
    for i in labels:
        print(f"    [{i}] → {emb['item_ids'][i]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    create_retrieval_pdf(
        output_path="results_example_fashionclip_joint_hnsw.pdf",
        query_item_id=emb["item_ids"][QUERY_INDEX],
        retrieved_item_ids=[emb["item_ids"][i] for i in labels],
        images_dir=IMAGES_DIR,
        distances=distances,
        retrieved_metadata=[aligned_labels[i] for i in labels],
        query_metadata=aligned_labels[QUERY_INDEX],
        result_dir=RESULTS_DIR,
    )
    print(f"  PDF   → {RESULTS_DIR}/results_example_fashionclip_joint_hnsw.pdf")

    print("\nGT evaluation...")
    gt_results = evaluate_i2i(
        embeddings=fused_np, hnsw=hnsw, ground_truth=gt, ks=EVAL_KS)
    for metric, vals in gt_results.items():
        line = "  ".join(f"@{k}: {v:.4f}" for k, v in vals.items())
        print(f"  {metric:<12} {line}")
    counts = [len(v) for v in gt.values() if v]
    print(f"  Avg/Med/Max relevant: "
          f"{np.mean(counts):.1f} / {np.median(counts):.1f} / {np.max(counts)}")

    print(f"\nAttribute profile (k={PROFILE_K})...")
    profile = evaluate_profile(
        embeddings=fused_np,
        search_method=hnsw,
        aligned_labels=aligned_labels,
        k=PROFILE_K,
    )
    print_profile_report(profile, method_name="fashionclip_joint",
                         backend_name="hnsw", k=PROFILE_K)

    if RUN_SIGNIFICANCE_TESTS:
        print(f"\n{'='*60}")
        print("STATISTICAL SIGNIFICANCE TESTS")
        print(f"{'='*60}")

        all_query_scores = {"fashionclip_joint": collect_query_scores(
            fused_np, hnsw, gt, ks=EVAL_KS)}

        for name, fname in COMPARISON_METHODS.items():
            path = f"{FUSED_EMBEDDINGS_DIR}/{fname}"
            if not os.path.exists(path):
                print(f"  [SKIP] {name}: embedding file not found ({path})")
                continue
            comp_np = np.load(path)
            comp_hnsw = HNSWSearch(dim=comp_np.shape[1])
            comp_hnsw.build(comp_np,
                            ef_construction=HNSW_EF_CONSTRUCTION,
                            M=HNSW_M,
                            ef_search=HNSW_EF_SEARCH)
            all_query_scores[name] = collect_query_scores(
                comp_np, comp_hnsw, gt, ks=EVAL_KS)
            print(f"  Loaded comparison: {name}  ({comp_np.shape[1]}-d)")

        sig_report = run_significance_tests(
            all_query_scores,
            ks=EVAL_KS,
            alpha=SIGNIFICANCE_ALPHA,
            baseline="fashionclip_joint",
        )
        print_significance_report(sig_report, alpha=SIGNIFICANCE_ALPHA)
        print_significance_summary_table(
            sig_report, ks=EVAL_KS, alpha=SIGNIFICANCE_ALPHA)
        print_effect_size_summary(sig_report, k=10)

    if RUN_SENSITIVITY:
        print(f"\n{'='*60}")
        print("GROUND TRUTH SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        run_sensitivity_analysis(
            embeddings_dict={"fashionclip_joint": fused_np},
            search_methods_dict={"fashionclip_joint": hnsw},
            aligned_labels=aligned_labels,
            ks=EVAL_KS,
            thresholds=SENSITIVITY_THRESHOLDS,
            metric=SENSITIVITY_METRIC,
            k=SENSITIVITY_K,
            run_perturbation=SENSITIVITY_N_PERTURB > 0,
            n_perturbation_samples=SENSITIVITY_N_PERTURB,
        )

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
        def flush(self):
            for s in self._streams:
                s.flush()

    txt_path = f"Dataset {dataset}/results_fashionclip_joint_runner.txt"
    os.makedirs(f"Dataset {dataset}", exist_ok=True)
    with open(txt_path, "w") as log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__
    print(f"Output saved to {txt_path}")
