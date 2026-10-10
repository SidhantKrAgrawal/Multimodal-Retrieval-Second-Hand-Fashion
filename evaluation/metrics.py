import numpy as np

def recall_at_k(retrieved, relevant, k):
    if len(relevant) == 0:
        return None
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def ndcg_at_k(retrieved, relevant, k):
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))

    return dcg / idcg if idcg > 0 else None

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if len(retrieved_k) == 0:
        return 0.0
    return len(set(retrieved_k) & relevant) / float(k)