import numpy as np
from evaluation.metrics import recall_at_k, ndcg_at_k, precision_at_k

def evaluate_i2i(embeddings, hnsw, ground_truth, ks=(5, 10, 20)):
    recall_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    precision_scores = {k: [] for k in ks}

    max_k = max(ks)

    for q in range(len(embeddings)):
        relevant = ground_truth[q]
        if len(relevant) == 0:
            continue

        query = embeddings[q:q+1]
        labels, _ = hnsw.search(query, k=max_k + 1)
        labels = np.asarray(labels)

        retrieved = labels[0]
        retrieved = retrieved[retrieved != q]

        for k in ks:
            r = recall_at_k(retrieved, relevant, k)
            n = ndcg_at_k(retrieved, relevant, k)
            p = precision_at_k(retrieved, relevant, k)

            if r is not None:
                recall_scores[k].append(r)
            if n is not None:
                ndcg_scores[k].append(n)
            precision_scores[k].append(p)

    return {
        "recall": {k: float(np.mean(recall_scores[k])) for k in ks},
        "ndcg": {k: float(np.mean(ndcg_scores[k])) for k in ks},
        "precision": {k: float(np.mean(precision_scores[k])) for k in ks},
    }
