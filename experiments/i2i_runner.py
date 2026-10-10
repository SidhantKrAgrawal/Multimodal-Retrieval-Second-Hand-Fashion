import os
import numpy as np

dataset              = os.environ.get("DATASET", "1")
EMBEDDINGS_DIR       = f"Dataset {dataset}/embeddings"
LABELS_DIR           = f"Dataset {dataset}/labels"
IMAGES_DIR           = f"Dataset {dataset}/images"
FUSED_EMBEDDINGS_DIR = f"Dataset {dataset}/fused_embeddings"
RESULTS_DIR          = f"Dataset {dataset}/results"

from load_embeddings import load_all_embeddings
from retrieval.hnsw_search import HNSWSearch
from retrieval.annoy_search import AnnoySearch
from retrieval.late_fusion import weighted_score_fusion
from visualization.pdf_results import create_retrieval_pdf
from evaluation.load_labels import load_all_labels
from evaluation.align_labels import align_labels_with_embeddings
from evaluation.build_ground_truth import build_ground_truth
from evaluation.evaluate_i2i import evaluate_i2i
from evaluation.evaluate_i2i_late_fusion import evaluate_i2i_late_fusion
from evaluation.evaluate_profile import (
    evaluate_profile, print_profile_report, compare_profiles,
    ATTRIBUTE_CONFIG, _attribute_match, _unique_values, _intra_list_diversity,
)
from evaluation.statistical_tests import (
    collect_query_scores,
    collect_query_scores_late_fusion,
    run_significance_tests,
    print_significance_report,
    print_significance_summary_table,
    print_effect_size_summary,
)
from evaluation.gt_sensitivity import run_sensitivity_analysis

EARLY_METHODS = [
    "selfattention",
    "coattention",
    "crossattention",
    "concat",
    "concat_text_heavy",
    "concat_image_heavy",
]

SEARCH_BACKENDS = [
    "hnsw",
    #"annoy",
]

QUERY_INDICES = {
    "selfattention":       999,
    "coattention":         999,
    "crossattention":      999,
    "concat":              999,
    "concat_text_heavy":   999,
    "concat_image_heavy":  999,
}

RUN_GT_EVALUATION = {
    "selfattention":       True,
    "coattention":         True,
    "crossattention":      True,
    "concat":              True,
    "concat_text_heavy":   True,
    "concat_image_heavy":  True,
}

RUN_PROFILE_EVALUATION = {
    "selfattention":       True,
    "coattention":         True,
    "crossattention":      True,
    "concat":              True,
    "concat_text_heavy":   True,
    "concat_image_heavy":  True,
}

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

LATE_FUSION_QUERY_IDX        = 4319
RUN_LATE_FUSION_GT_EVAL      = True
RUN_LATE_FUSION_PROFILE_EVAL = True
RUN_LATE_FUSION_SENSITIVITY  = True

EVAL_KS       = (5, 10, 20)
PROFILE_K     = 10
ANNOY_N_TREES = 50

COMPARISON_BACKEND = "hnsw"

RUN_COMPARISON         = True
RUN_SIGNIFICANCE_TESTS = True
SIGNIFICANCE_ALPHA     = 0.05
SIGNIFICANCE_BASELINE  = None   # None = all pairwise; "concat" or "late_fusion" = vs baseline

RUN_SENSITIVITY        = True
SENSITIVITY_METRIC     = "precision"
SENSITIVITY_K          = 10
SENSITIVITY_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
SENSITIVITY_N_PERTURB  = 20

def build_indexes(fused_np: np.ndarray) -> dict:
    dim     = fused_np.shape[1]
    indexes = {}

    if "hnsw" in SEARCH_BACKENDS:
        idx = HNSWSearch(dim=dim)
        idx.build(fused_np)
        indexes["hnsw"] = idx

    if "annoy" in SEARCH_BACKENDS:
        idx = AnnoySearch(dim=dim)
        idx.build(fused_np, n_trees=ANNOY_N_TREES)
        indexes["annoy"] = idx

    return indexes


def _dim_check(method: str, fused_np: np.ndarray):
    dim = fused_np.shape[1]
    expected = {
        "concat":            2432,
        "concat_text_heavy": 2432,
        "concat_image_heavy":2432,
        "concat_dino_boost": 2432,
        "selfattention":     2048,
        "coattention":       2048,
        "crossattention":    2048,
    }
    exp = expected.get(method, "?")
    flag = "" if dim == exp else f"  ⚠ expected {exp}"
    print(f"  Embedding dim: {dim}{flag}")


def run_early_search(search_method, fused_np, query_idx,
                     emb, aligned_labels, method_name, backend_name):
    query = fused_np[query_idx:query_idx + 1]
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

    output_pdf = f"results_example_{method_name}_{backend_name}.pdf"
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


def run_gt_evaluation(fused_np, search_method, gt):
    results = evaluate_i2i(
        embeddings=fused_np, hnsw=search_method,
        ground_truth=gt, ks=EVAL_KS,
    )
    print("  GT results:")
    for metric, vals in results.items():
        line = "  ".join(f"@{k}: {v:.4f}" for k, v in vals.items())
        print(f"    {metric:<12} {line}")
    counts = [len(v) for v in gt.values() if v]
    print(f"    Avg/Med/Max relevant: "
          f"{np.mean(counts):.1f} / {np.median(counts):.1f} / {np.max(counts)}")


def build_late_fusion_indexes(raw_embeddings: dict) -> dict:
    per_mod = {}
    for m, arr in raw_embeddings.items():
        idx = HNSWSearch(dim=arr.shape[1])
        idx.build(arr)
        per_mod[m] = idx
    print(f"  Built HNSW indexes: {list(per_mod.keys())}")
    return per_mod


class LateFusionSearchAdapter:
    """Wraps late fusion to satisfy the search_method.search(query_vec, k) interface."""

    def __init__(self, raw_embeddings, per_mod_indexes, weights,
                 alpha, search_k, n_items):
        self._raw       = raw_embeddings
        self._indexes   = per_mod_indexes
        self._weights   = weights
        self._alpha     = alpha
        self._search_k  = search_k
        self._n         = n_items
        first_mod       = next(iter(raw_embeddings))
        self._ref_emb   = raw_embeddings[first_mod]   # (N, D_first)

    def search(self, query_vec, k):
        # recover item index from the passed row slice to query each modality separately
        query_vec = np.asarray(query_vec)
        diffs     = np.linalg.norm(self._ref_emb - query_vec, axis=1)
        q_idx     = int(np.argmin(diffs))

        results_buffer = {}
        for m, idx in self._indexes.items():
            q_vec = self._raw[m][q_idx:q_idx + 1]
            lbl, dist = idx.search(q_vec, k=self._search_k + 1)
            lbl  = np.asarray(lbl[0])
            dist = np.asarray(dist[0])
            mask = lbl != q_idx
            results_buffer[m] = (lbl[mask], dist[mask])

        fused  = weighted_score_fusion(results_buffer, self._weights,
                                       alpha=self._alpha)
        labels = np.array([i for i, _ in fused[:k]])
        scores = np.array([s for _, s in fused[:k]])
        return labels[np.newaxis, :], scores[np.newaxis, :]


def run_late_fusion_search(raw_embeddings, per_mod_indexes,
                           query_idx, emb, aligned_labels):
    results_buffer = {}
    for m, idx in per_mod_indexes.items():
        q = raw_embeddings[m][query_idx:query_idx + 1]
        labels, distances = idx.search(q, k=LATE_FUSION_SEARCH_K + 1)
        labels    = np.asarray(labels[0])
        distances = np.asarray(distances[0])
        mask = labels != query_idx
        results_buffer[m] = (labels[mask], distances[mask])

    fused = weighted_score_fusion(
        results_buffer, LATE_FUSION_WEIGHTS, alpha=LATE_FUSION_ALPHA)[:10]

    print(f"\n  Query : {emb['item_ids'][query_idx]}")
    print("  Late-fusion retrieved:")
    for item_idx, score in fused:
        print(f"    [{item_idx}] → {emb['item_ids'][item_idx]}  "
              f"score: {score:.5f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    create_retrieval_pdf(
        output_path="results_example_latefusion.pdf",
        query_item_id=emb["item_ids"][query_idx],
        retrieved_item_ids=[emb["item_ids"][i] for i, _ in fused],
        images_dir=IMAGES_DIR,
        distances=[s for _, s in fused],
        retrieved_metadata=[aligned_labels[i] for i, _ in fused],
        query_metadata=aligned_labels[query_idx],
        result_dir=RESULTS_DIR,
    )
    print(f"  PDF   → {RESULTS_DIR}/results_example_latefusion.pdf")


def run_late_fusion_gt_eval(raw_embeddings, per_mod_indexes, gt):
    results = evaluate_i2i_late_fusion(
        embeddings=raw_embeddings,
        indices=per_mod_indexes,
        weights=LATE_FUSION_WEIGHTS,
        ground_truth=gt,
        ks=EVAL_KS,
        fusion_fn=weighted_score_fusion,
        alpha=LATE_FUSION_ALPHA,
        search_k=LATE_FUSION_SEARCH_K,
    )
    print("  GT results:")
    for metric, vals in results.items():
        line = "  ".join(f"@{k}: {v:.4f}" for k, v in vals.items())
        print(f"    {metric:<12} {line}")


def run_late_fusion_profile(raw_embeddings, per_mod_indexes, aligned_labels):
    from collections import defaultdict as _dd

    focus_accum   = _dd(list)
    variety_accum = _dd(list)
    ild_list      = []
    per_query     = []
    n             = len(next(iter(raw_embeddings.values())))

    for q_idx in range(n):
        results_buffer = {}
        for m, idx in per_mod_indexes.items():
            q_vec = raw_embeddings[m][q_idx:q_idx + 1]
            labels, distances = idx.search(q_vec, k=LATE_FUSION_SEARCH_K + 1)
            labels    = np.asarray(labels[0])
            distances = np.asarray(distances[0])
            mask = labels != q_idx
            results_buffer[m] = (labels[mask], distances[mask])

        fused     = weighted_score_fusion(
            results_buffer, LATE_FUSION_WEIGHTS, alpha=LATE_FUSION_ALPHA)
        top_k_ids = [i for i, _ in fused[:PROFILE_K]]
        q_meta    = aligned_labels[q_idx]
        r_metas   = [aligned_labels[i] for i in top_k_ids]

        q_focus = {}
        for attr in ATTRIBUTE_CONFIG:
            vals = [_attribute_match(attr, q_meta.get(attr), r.get(attr))
                    for r in r_metas]
            vals = [v for v in vals if v is not None]
            rate = float(np.mean(vals)) if vals else None
            q_focus[attr] = rate
            if rate is not None:
                focus_accum[attr].append(rate)

        q_variety = {}
        for attr in ATTRIBUTE_CONFIG:
            n_u = _unique_values(attr, r_metas)
            q_variety[attr] = n_u
            variety_accum[attr].append(n_u)

        ild = _intra_list_diversity(r_metas)
        ild_list.append(ild)
        per_query.append({
            "query_idx": q_idx, "query_meta": q_meta,
            "retrieved": top_k_ids, "focus": q_focus,
            "variety": q_variety, "ild": ild,
        })

    return {
        "focus":    {a: float(np.mean(v)) if v else None
                     for a, v in focus_accum.items()},
        "variety":  {a: float(np.mean(v)) if v else None
                     for a, v in variety_accum.items()},
        "ild":      float(np.mean(ild_list)) if ild_list else None,
        "per_query": per_query,
    }


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

    for method in EARLY_METHODS:
        print(f"\n{'='*60}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*60}")

        emb_file = f"{FUSED_EMBEDDINGS_DIR}/{EMBEDDING_FILES[method]}"

        # Skip gracefully if file not generated yet
        if not os.path.exists(emb_file):
            print(f"  [SKIP] Embedding file not found: {emb_file}")
            print(f"  Run the corresponding training/generation script first.")
            continue

        fused_np = np.load(emb_file)
        _dim_check(method, fused_np)

        indexes   = build_indexes(fused_np)
        query_idx = QUERY_INDICES[method]

        for backend_name, search_method in indexes.items():
            print(f"\n  --- Backend: {backend_name.upper()} ---")

            run_early_search(search_method, fused_np, query_idx,
                             emb, aligned_labels, method, backend_name)

            if RUN_GT_EVALUATION.get(method, False):
                run_gt_evaluation(fused_np, search_method, gt)
            else:
                print("  (GT evaluation skipped)")

            if backend_name == COMPARISON_BACKEND:
                if RUN_SIGNIFICANCE_TESTS:
                    print("  Collecting per-query scores...")
                    all_query_scores[method] = collect_query_scores(
                        fused_np, search_method, gt, ks=EVAL_KS)

                if RUN_SENSITIVITY:
                    all_embeddings_for_sensitivity[method]     = fused_np
                    all_search_methods_for_sensitivity[method] = search_method

            if RUN_PROFILE_EVALUATION.get(method, False):
                profile = evaluate_profile(
                    embeddings=fused_np,
                    search_method=search_method,
                    aligned_labels=aligned_labels,
                    k=PROFILE_K,
                )
                print_profile_report(profile, method_name=method,
                                     backend_name=backend_name, k=PROFILE_K)
                if backend_name == COMPARISON_BACKEND:
                    all_profiles[method] = profile
            else:
                print("  (Profile evaluation skipped)")

    if RUN_LATE_FUSION:
        print(f"\n{'='*60}")
        print("METHOD: LATE FUSION (Weighted Exponential Score)")
        print(f"  Modalities : {LATE_FUSION_MODALITIES}")
        print(f"  Weights    : {LATE_FUSION_WEIGHTS}")
        print(f"  Alpha      : {LATE_FUSION_ALPHA}  |  "
              f"Search-k: {LATE_FUSION_SEARCH_K}")
        print(f"{'='*60}")

        raw_embeddings = {
            m: emb[m].numpy().astype(np.float32)
            for m in LATE_FUSION_MODALITIES
        }
        per_mod_indexes = build_late_fusion_indexes(raw_embeddings)

        run_late_fusion_search(raw_embeddings, per_mod_indexes,
                               LATE_FUSION_QUERY_IDX, emb, aligned_labels)

        if RUN_LATE_FUSION_GT_EVAL:
            print("\n  Running GT evaluation...")
            run_late_fusion_gt_eval(raw_embeddings, per_mod_indexes, gt)

            if RUN_SIGNIFICANCE_TESTS:
                print("  Collecting per-query scores for significance tests...")
                all_query_scores["late_fusion"] = \
                    collect_query_scores_late_fusion(
                        raw_embeddings, per_mod_indexes,
                        LATE_FUSION_WEIGHTS, gt,
                        ks=EVAL_KS,
                        alpha=LATE_FUSION_ALPHA,
                        search_k=LATE_FUSION_SEARCH_K,
                    )

        if RUN_LATE_FUSION_PROFILE_EVAL:
            print("\n  Running profile evaluation...")
            lf_profile = run_late_fusion_profile(
                raw_embeddings, per_mod_indexes, aligned_labels)
            print_profile_report(lf_profile,
                                 method_name="late_fusion",
                                 backend_name="hnsw",
                                 k=PROFILE_K)
            all_profiles["late_fusion"] = lf_profile

        if RUN_SENSITIVITY and RUN_LATE_FUSION_SENSITIVITY:
            print("\n  Building LateFusionSearchAdapter for sensitivity analysis...")
            n_items = len(next(iter(raw_embeddings.values())))
            lf_adapter = LateFusionSearchAdapter(
                raw_embeddings  = raw_embeddings,
                per_mod_indexes = per_mod_indexes,
                weights         = LATE_FUSION_WEIGHTS,
                alpha           = LATE_FUSION_ALPHA,
                search_k        = LATE_FUSION_SEARCH_K,
                n_items         = n_items,
            )
            # reference array used only for query-index recovery inside the adapter
            first_mod = next(iter(raw_embeddings))
            all_embeddings_for_sensitivity["late_fusion"]     = raw_embeddings[first_mod]
            all_search_methods_for_sensitivity["late_fusion"] = lf_adapter

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

    txt_path = f"Dataset {dataset}/results_i2i_runner.txt"
    os.makedirs(f"Dataset {dataset}", exist_ok=True)
    with open(txt_path, "w") as log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__
    print(f"Output saved to {txt_path}")