import numpy as np
from evaluation.metrics import recall_at_k, ndcg_at_k, precision_at_k
from retrieval.late_fusion import weighted_score_fusion


def evaluate_i2i_late_fusion(
    embeddings,
    indices,
    weights,
    ground_truth,
    ks=(5, 10, 20),
    fusion_fn=weighted_score_fusion,
    alpha=5.0,
    search_k=None,
):
    if search_k is None:
        search_k = max(ks) * 3

    recall_scores    = {k: [] for k in ks}
    ndcg_scores      = {k: [] for k in ks}
    precision_scores = {k: [] for k in ks}

    num_queries = len(next(iter(embeddings.values())))

    for q in range(num_queries):
        relevant = ground_truth[q]
        if not relevant:
            continue

        results = {}

        for m, index in indices.items():
            query_vec = embeddings[m][q:q + 1]
            labels, distances = index.search(query_vec, k=search_k + 1)

            labels    = np.asarray(labels)
            distances = np.asarray(distances)
            if labels.ndim > 1:
                labels, distances = labels[0], distances[0]

            mask = labels != q
            results[m] = (labels[mask], distances[mask])

        fused_ranking = fusion_fn(results, weights, alpha=alpha)
        ranked_ids    = [idx for idx, _ in fused_ranking]

        for k in ks:
            r = recall_at_k(ranked_ids, relevant, k)
            n = ndcg_at_k(ranked_ids, relevant, k)
            p = precision_at_k(ranked_ids, relevant, k)

            if r is not None:
                recall_scores[k].append(r)
            if n is not None:
                ndcg_scores[k].append(n)
            if p is not None:
                precision_scores[k].append(p)

    return {
        "recall":    {k: float(np.mean(v)) if v else 0.0 for k, v in recall_scores.items()},
        "ndcg":      {k: float(np.mean(v)) if v else 0.0 for k, v in ndcg_scores.items()},
        "precision": {k: float(np.mean(v)) if v else 0.0 for k, v in precision_scores.items()},
    }