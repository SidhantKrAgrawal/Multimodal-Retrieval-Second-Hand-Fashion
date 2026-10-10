import os
import numpy as np

dataset              = os.environ.get("DATASET", "1")
EMBEDDINGS_DIR       = f"Dataset {dataset}/embeddings"
LABELS_DIR           = f"Dataset {dataset}/labels"
IMAGES_DIR           = f"Dataset {dataset}/images"
RESULTS_DIR          = f"Dataset {dataset}/results"

from load_embeddings import load_all_embeddings
from retrieval.hnsw_search import HNSWSearch
from visualization.pdf_results import create_retrieval_pdf
from evaluation.load_labels import load_all_labels
from evaluation.align_labels import align_labels_with_embeddings
from evaluation.build_ground_truth import build_ground_truth
from evaluation.evaluate_i2i import evaluate_i2i
from evaluation.evaluate_profile import (
    evaluate_profile, print_profile_report, compare_profiles,
)
from evaluation.statistical_tests import (
    collect_query_scores,
    run_significance_tests,
    print_significance_report,
    print_significance_summary_table,
    print_effect_size_summary,
)
from evaluation.gt_sensitivity import run_sensitivity_analysis

RAW_MODALITIES = [
    "image_clip",
    "image_dino",
    "text_clip",
    "text_qwen",
]

QUERY_INDEX = 999
EVAL_KS    = (5, 10, 20)
PROFILE_K  = 10

RUN_GT_EVALUATION      = True
RUN_PROFILE_EVALUATION = True
RUN_COMPARISON         = True

RUN_SIGNIFICANCE_TESTS = True
SIGNIFICANCE_ALPHA     = 0.05
SIGNIFICANCE_BASELINE  = None

RUN_SENSITIVITY        = True
SENSITIVITY_METRIC     = "precision"
SENSITIVITY_K          = 10
SENSITIVITY_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
SENSITIVITY_N_PERTURB  = 20

def build_index(emb_np: np.ndarray) -> HNSWSearch:
    idx = HNSWSearch(dim=emb_np.shape[1])
    idx.build(emb_np)
    return idx


def run_example_search(emb_np, search_method, query_idx,
                       emb, aligned_labels, modality_name):
    query = emb_np[query_idx:query_idx + 1]
    labels, distances = search_method.search(query, k=11)
    labels    = np.asarray(labels)
    distances = np.asarray(distances)
    if labels.ndim > 1:
        labels, distances = labels[0], distances[0]
    mask      = labels != query_idx
    labels    = labels[mask][:10]
    distances = distances[mask][:10]

    print(f"\n  Query : {emb['item_ids'][query_idx]}")
    print("  Retrieved:")
    for i in labels:
        print(f"    [{i}] → {emb['item_ids'][i]}")

    output_pdf = f"results_example_{modality_name}_hnsw.pdf"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    create_retrieval_pdf(
        output_path=output_pdf,
        query_item_id=emb["item_ids"][query_idx],
        retrieved_item_ids=[emb["item_ids"][i] for i in labels],
        images_dir=IMAGES_DIR,
        distances=distances,
        retrieved_metadata=[aligned_labels[i] for i in labels],
        query_metadata=aligned_labels[query_idx],
        result_dir=RESULTS_DIR,
    )
    print(f"  PDF   → {RESULTS_DIR}/{output_pdf}")


def run_gt_evaluation(emb_np, search_method, gt):
    results = evaluate_i2i(
        embeddings=emb_np, hnsw=search_method,
        ground_truth=gt, ks=EVAL_KS,
    )
    print("  GT results:")
    for metric, vals in results.items():
        line = "  ".join(f"@{k}: {v:.4f}" for k, v in vals.items())
        print(f"    {metric:<12} {line}")
    counts = [len(v) for v in gt.values() if v]
    print(f"    Avg/Med/Max relevant: "
          f"{np.mean(counts):.1f} / {np.median(counts):.1f} / {np.max(counts)}")


def main():
    print("Loading embeddings and labels...")
    emb            = load_all_embeddings(EMBEDDINGS_DIR)
    labels_dict    = load_all_labels(LABELS_DIR)
    aligned_labels = align_labels_with_embeddings(emb["item_ids"], labels_dict)
    gt             = build_ground_truth(aligned_labels)

    all_profiles                       = {}
    all_query_scores                   = {}
    all_embeddings_for_sensitivity     = {}
    all_search_methods_for_sensitivity = {}

    for modality in RAW_MODALITIES:
        print(f"\n{'='*60}")
        print(f"MODALITY: {modality.upper()}")
        print(f"{'='*60}")

        emb_np = emb[modality].numpy().astype(np.float32)
        print(f"  Embedding dim: {emb_np.shape[1]}")

        search_method = build_index(emb_np)

        run_example_search(emb_np, search_method, QUERY_INDEX,
                           emb, aligned_labels, modality)

        if RUN_GT_EVALUATION:
            run_gt_evaluation(emb_np, search_method, gt)

        if RUN_SIGNIFICANCE_TESTS:
            print("  Collecting per-query scores...")
            all_query_scores[modality] = collect_query_scores(
                emb_np, search_method, gt, ks=EVAL_KS)

        if RUN_SENSITIVITY:
            all_embeddings_for_sensitivity[modality]     = emb_np
            all_search_methods_for_sensitivity[modality] = search_method

        if RUN_PROFILE_EVALUATION:
            profile = evaluate_profile(
                embeddings=emb_np,
                search_method=search_method,
                aligned_labels=aligned_labels,
                k=PROFILE_K,
            )
            print_profile_report(profile, method_name=modality,
                                 backend_name="hnsw", k=PROFILE_K)
            all_profiles[modality] = profile

    if RUN_COMPARISON and len(all_profiles) > 1:
        compare_profiles(all_profiles, k=PROFILE_K)

    if RUN_SIGNIFICANCE_TESTS and len(all_query_scores) > 1:
        print(f"\n{'='*60}")
        print("STATISTICAL SIGNIFICANCE TESTS")
        print(f"{'='*60}")
        print(f"  Methods in test: {list(all_query_scores.keys())}")

        sig_report = run_significance_tests(
            all_query_scores,
            ks=EVAL_KS,
            alpha=SIGNIFICANCE_ALPHA,
            baseline=SIGNIFICANCE_BASELINE,
        )
        print_significance_report(sig_report, alpha=SIGNIFICANCE_ALPHA)
        print_significance_summary_table(
            sig_report, ks=EVAL_KS, alpha=SIGNIFICANCE_ALPHA)
        print_effect_size_summary(sig_report, k=10)

    if RUN_SENSITIVITY and all_embeddings_for_sensitivity:
        print(f"\n{'='*60}")
        print("GROUND TRUTH SENSITIVITY ANALYSIS")
        print(f"  Methods: {list(all_embeddings_for_sensitivity.keys())}")
        print(f"{'='*60}")
        run_sensitivity_analysis(
            embeddings_dict=all_embeddings_for_sensitivity,
            search_methods_dict=all_search_methods_for_sensitivity,
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
    import sys

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
        def flush(self):
            for s in self._streams:
                s.flush()

    txt_path = f"Dataset {dataset}/results_i2i_raw_runner.txt"
    os.makedirs(f"Dataset {dataset}", exist_ok=True)
    with open(txt_path, "w") as log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__
    print(f"Output saved to {txt_path}")
