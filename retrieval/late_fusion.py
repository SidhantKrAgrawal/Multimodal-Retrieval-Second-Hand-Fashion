import numpy as np
from collections import defaultdict


def weighted_score_fusion(results, weights, alpha=5.0):
    """
    Score fusion via exponential decay (exp(-alpha * distance)).
    Maps distance=0 → score=1.0, distance=inf → score=0.0 smoothly.
    """
    fused_scores = defaultdict(float)

    for modality, (indices, distances) in results.items():
        w = weights.get(modality, 1.0)
        distances = np.array(distances)
        scores = np.exp(-alpha * distances)

        for idx, s in zip(indices, scores):
            fused_scores[idx] += w * s

    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked