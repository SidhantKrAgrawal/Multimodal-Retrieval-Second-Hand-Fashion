import os
import sys
import random
import numpy as np

dataset              = os.environ.get("DATASET", "1")
EMBEDDINGS_DIR       = f"Dataset {dataset}/embeddings"
LABELS_DIR           = f"Dataset {dataset}/labels"
IMAGES_DIR           = f"Dataset {dataset}/images"
FUSED_EMBEDDINGS_DIR = f"Dataset {dataset}/fused_embeddings"
RESULTS_DIR          = f"Dataset {dataset}/results"

from load_embeddings import load_all_embeddings
from evaluation.load_labels import load_all_labels
from evaluation.align_labels import align_labels_with_embeddings
from evaluation.build_ground_truth import build_ground_truth
from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k
from evaluation.evaluate_profile import (
    evaluate_profile, print_profile_report, compare_profiles,
    _attribute_match, _unique_values, _intra_list_diversity,
    ATTRIBUTE_CONFIG,
)
from evaluation.statistical_tests import (
    run_significance_tests,
    print_significance_report,
    print_significance_summary_table,
    print_effect_size_summary,
)
from retrieval.bm25_search import BM25Corpus, TFIDFCorpus
from retrieval.hnsw_search import HNSWSearch
from visualization.pdf_results import create_retrieval_pdf

EVAL_KS    = (5, 10, 20)
PROFILE_K  = 10
QUERY_INDEX = 999

RUN_SIGNIFICANCE = True
SIGNIFICANCE_ALPHA = 0.05

COMPARISON_EMBEDDINGS = {
    "selfattention":    "fused_item_embeddings_selfattention.npy",
    "concat_text_heavy":"fused_item_embeddings_concat_text_heavy.npy",
    "concat":           "fused_item_embeddings_concat.npy",
}

HNSW_M               = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH       = 50

def _print_sep(title="", width=60):
    if title:
        print(f"\n{'='*width}")
        print(f"METHOD: {title}")
        print(f"{'='*width}")
    else:
        print(f"\n{'='*width}")


def evaluate_text_method(name, searcher, aligned_labels, ground_truth,
                          item_ids, ks=EVAL_KS, profile_k=PROFILE_K,
                          query_idx=QUERY_INDEX):
    max_k = max(ks)

    results = searcher.search(query_idx, k=max_k + 1)
    retrieved_ids = [idx for idx, _ in results[:10]]
    distances     = [1.0 / (sc + 1e-9) for _, sc in results[:10]]

    print(f"  Query : {item_ids[query_idx]}")
    print("  Retrieved:")
    for idx, sc in results[:10]:
        print(f"    [{idx}] → {item_ids[idx]}  score: {sc:.5f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pdf_path = f"results_example_{name}.pdf"
    try:
        create_retrieval_pdf(
            output_path=pdf_path,
            query_item_id=item_ids[query_idx],
            retrieved_item_ids=[item_ids[i] for i in retrieved_ids],
            images_dir=IMAGES_DIR,
            distances=distances,
            retrieved_metadata=[aligned_labels[i] for i in retrieved_ids],
            query_metadata=aligned_labels[query_idx],
            result_dir=RESULTS_DIR,
        )
        print(f"  PDF   → {RESULTS_DIR}/{pdf_path}")
    except Exception as e:
        print(f"  [PDF skipped: {e}]")

    recall_scores    = {k: [] for k in ks}
    ndcg_scores      = {k: [] for k in ks}
    precision_scores = {k: [] for k in ks}
    per_query_scores = {
        "precision": {k: [] for k in ks},
        "recall":    {k: [] for k in ks},
        "ndcg":      {k: [] for k in ks},
    }

    n_items = len(aligned_labels)

    for q in range(n_items):
        relevant = ground_truth[q]
        if not relevant:
            continue

        results_q = searcher.search(q, k=max_k + 1)
        retrieved  = np.array([idx for idx, _ in results_q if idx != q])

        for k in ks:
            p = precision_at_k(retrieved, relevant, k)
            r = recall_at_k(retrieved, relevant, k)
            n = ndcg_at_k(retrieved, relevant, k)
            precision_scores[k].append(p)
            per_query_scores["precision"][k].append(p)
            if r is not None:
                recall_scores[k].append(r)
                per_query_scores["recall"][k].append(r)
            if n is not None:
                ndcg_scores[k].append(n)
                per_query_scores["ndcg"][k].append(n)

    gt_results = {
        "recall":    {k: float(np.mean(recall_scores[k]))    for k in ks},
        "ndcg":      {k: float(np.mean(ndcg_scores[k]))      for k in ks},
        "precision": {k: float(np.mean(precision_scores[k])) for k in ks},
    }

    print("  GT results:")
    for metric, vals in gt_results.items():
        line = "  ".join(f"@{k}: {v:.4f}" for k, v in sorted(vals.items()))
        print(f"    {metric:<12} {line}")

    counts = [len(v) for v in ground_truth.values() if v]
    print(f"  Avg/Med/Max relevant: "
          f"{np.mean(counts):.1f} / {np.median(counts):.1f} / {np.max(counts)}")

    focus_sums    = {attr: [] for attr in ATTRIBUTE_CONFIG}
    variety_sums  = {attr: [] for attr in ATTRIBUTE_CONFIG}
    ild_scores    = []

    for q in range(n_items):
        q_label = aligned_labels[q]
        if not q_label:
            continue

        results_q  = searcher.search(q, k=profile_k + 1)
        ret_indices = [idx for idx, _ in results_q if idx != q][:profile_k]
        if not ret_indices:
            continue

        ret_labels = [aligned_labels[i] for i in ret_indices]

        for attr in ATTRIBUTE_CONFIG:
            q_val = q_label.get(attr)
            matches = []
            for rl in ret_labels:
                m = _attribute_match(attr, q_val, rl.get(attr))
                if m is not None:
                    matches.append(m)
            if matches:
                focus_sums[attr].append(float(np.mean(matches)))

            variety_sums[attr].append(_unique_values(attr, ret_labels))

        ild_scores.append(_intra_list_diversity(ret_labels))

    profile = {
        "focus":   {a: float(np.mean(v)) if v else 0.0 for a, v in focus_sums.items()},
        "variety": {a: float(np.mean(v)) if v else 0.0 for a, v in variety_sums.items()},
        "ild":     float(np.mean(ild_scores)) if ild_scores else 0.0,
    }

    print_profile_report(profile, method_name=name, backend_name="text", k=profile_k)

    per_query_np = {
        metric: {k: np.array(v) for k, v in ks_dict.items()}
        for metric, ks_dict in per_query_scores.items()
    }

    return gt_results, profile, per_query_np


class RandomSearch:

    def __init__(self, n_items: int, seed: int = 42):
        self._n = n_items
        self._rng = random.Random(seed)

    def search(self, query_idx: int, k: int) -> list[tuple[int, float]]:
        indices = list(range(self._n))
        indices.remove(query_idx)
        self._rng.shuffle(indices)
        return [(idx, 1.0) for idx in indices[:k]]


def main():
    print("Loading embeddings and labels...")
    emb            = load_all_embeddings(EMBEDDINGS_DIR)
    item_ids       = emb["item_ids"]
    labels_dict    = load_all_labels(LABELS_DIR)
    aligned_labels = align_labels_with_embeddings(item_ids, labels_dict)
    gt             = build_ground_truth(aligned_labels)
    n_items        = len(item_ids)

    print(f"  Catalogue size : {n_items} items")
    print(f"  Queries with GT: {sum(1 for v in gt.values() if v)}")

    all_query_scores = {}

    _print_sep("BM25  (Robertson & Zaragoza, 2009)  k1=1.5  b=0.75")
    print("  Building BM25 index from label text...")
    bm25 = BM25Corpus(aligned_labels, k1=1.5, b=0.75)
    vocab_size = len(bm25._inv_idx)
    avg_doc_len = bm25._avgdl
    print(f"  Vocabulary size : {vocab_size:,} terms")
    print(f"  Avg document len: {avg_doc_len:.1f} tokens")

    gt_bm25, profile_bm25, qs_bm25 = evaluate_text_method(
        "bm25", bm25, aligned_labels, gt, item_ids)
    all_query_scores["bm25"] = qs_bm25

    _print_sep("TF-IDF  (log-TF, cosine similarity, scikit-learn)")
    print("  Building TF-IDF matrix from label text...")
    tfidf = TFIDFCorpus(aligned_labels)
    shape = tfidf._matrix.shape
    print(f"  Matrix shape    : {shape[0]} items × {shape[1]} vocab")

    gt_tfidf, profile_tfidf, qs_tfidf = evaluate_text_method(
        "tfidf", tfidf, aligned_labels, gt, item_ids)
    all_query_scores["tfidf"] = qs_tfidf

    _print_sep("RANDOM  (uniform random, seed=42)  — lower bound")
    rand = RandomSearch(n_items=n_items, seed=42)

    gt_rand, profile_rand, qs_rand = evaluate_text_method(
        "random", rand, aligned_labels, gt, item_ids)
    all_query_scores["random"] = qs_rand

    if RUN_SIGNIFICANCE:
        print(f"\n{'='*60}")
        print("LOADING EMBEDDING COMPARISONS FOR SIGNIFICANCE TESTS")
        print(f"{'='*60}")

        from retrieval.hnsw_search import HNSWSearch
        from evaluation.statistical_tests import collect_query_scores

        for cname, fname in COMPARISON_EMBEDDINGS.items():
            fpath = f"{FUSED_EMBEDDINGS_DIR}/{fname}"
            if not os.path.exists(fpath):
                print(f"  [SKIP] {cname}: {fpath} not found")
                continue
            comp_np = np.load(fpath)
            comp_hnsw = HNSWSearch(dim=comp_np.shape[1])
            comp_hnsw.build(comp_np,
                            ef_construction=HNSW_EF_CONSTRUCTION,
                            M=HNSW_M,
                            ef_search=HNSW_EF_SEARCH)
            all_query_scores[cname] = collect_query_scores(
                comp_np, comp_hnsw, gt, ks=EVAL_KS)
            print(f"  Loaded: {cname}  ({comp_np.shape[1]}-d)")

    if RUN_SIGNIFICANCE and len(all_query_scores) > 1:
        print(f"\n{'='*60}")
        print("STATISTICAL SIGNIFICANCE TESTS  (baseline: bm25)")
        print(f"{'='*60}")

        sig_bm25 = run_significance_tests(
            all_query_scores, ks=EVAL_KS,
            alpha=SIGNIFICANCE_ALPHA, baseline="bm25",
        )
        print_significance_report(sig_bm25, alpha=SIGNIFICANCE_ALPHA)
        print_significance_summary_table(sig_bm25, ks=EVAL_KS, alpha=SIGNIFICANCE_ALPHA)
        print_effect_size_summary(sig_bm25, k=10)

    print(f"\n{'='*70}")
    print("SUMMARY TABLE  (precision@K  |  dataset = {})".format(dataset))
    print(f"{'='*70}")
    header = f"{'Method':<22} {'P@5':>7} {'P@10':>7} {'P@20':>7} {'NDCG@10':>9} {'R@10':>7} {'ILD':>7}"
    print(header)
    print("-" * 70)

    rows = [
        ("bm25",    gt_bm25,   profile_bm25),
        ("tfidf",   gt_tfidf,  profile_tfidf),
        ("random",  gt_rand,   profile_rand),
    ]
    for name, gt_res, prof in rows:
        p5   = gt_res["precision"].get(5,  0.0)
        p10  = gt_res["precision"].get(10, 0.0)
        p20  = gt_res["precision"].get(20, 0.0)
        n10  = gt_res["ndcg"].get(10,      0.0)
        r10  = gt_res["recall"].get(10,    0.0)
        ild  = prof["ild"]
        print(f"  {name:<20} {p5:>7.4f} {p10:>7.4f} {p20:>7.4f} {n10:>9.4f} {r10:>7.4f} {ild:>7.4f}")

    print(f"\n{'='*70}")
    print("FOCUS COMPARISON  (category / type match rate at k=10)")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'category':>10} {'type':>10} {'colors':>10} {'pattern':>10}")
    for name, _, prof in rows:
        f = prof["focus"]
        print(f"  {name:<20} {f.get('category',0):>10.3f} {f.get('type',0):>10.3f} "
              f"{f.get('colors',0):>10.3f} {f.get('pattern',0):>10.3f}")

    print(f"\nDone.  Results saved to Dataset {dataset}/results_text_baselines_runner.txt")


if __name__ == "__main__":
    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams: s.write(data)
        def flush(self):
            for s in self._streams: s.flush()

    os.makedirs(f"Dataset {dataset}", exist_ok=True)
    txt_path = f"Dataset {dataset}/results_text_baselines_runner.txt"
    with open(txt_path, "w") as log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__
    print(f"Output saved to {txt_path}")
