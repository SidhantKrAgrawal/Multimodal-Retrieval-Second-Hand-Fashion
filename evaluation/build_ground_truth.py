from collections import defaultdict

def item_similarity(a, b):
    score = 0.0

    if not a or not b:
        return 0.0

    cat_a = a.get("category")
    cat_b = b.get("category")
    if cat_a and cat_b and cat_a == cat_b:
        score += 0.35

    if a.get("type") == b.get("type") and a.get("type") is not None:
        score += 0.25

    colors_a = set(a.get("colors", []))
    colors_b = set(b.get("colors", []))
    if colors_a & colors_b:
        score += 0.15

    season_a = a.get("season") or ""
    season_b = b.get("season") or ""
    if season_a and season_b and season_a == season_b:
        score += 0.1

    brand_a = a.get("brand") or ""
    brand_b = b.get("brand") or ""
    if brand_a and brand_b and brand_a == brand_b:
        score += 0.05

    return score

def build_ground_truth(labels, threshold=0.7):
    gt = defaultdict(set)

    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i == j:
                continue
            if item_similarity(li, lj) >= threshold:
                gt[i].add(j)

    return gt