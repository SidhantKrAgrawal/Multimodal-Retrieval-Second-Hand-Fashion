import numpy as np
from collections import Counter, defaultdict

ATTRIBUTE_CONFIG = {
    "category": "scalar",
    "type":     "scalar",
    "colors":   "list",
    "season":   "scalar",
    "brand":    "scalar",
    "pattern":  "scalar",
    "cut":      "list",
    "price":    "scalar",
    "condition":"scalar",
}


def _scalar_match(val_q, val_r):
    if val_q is None or val_r is None:
        return None  # excluded from mean — avoids penalising missing fields
    return int(val_q == val_r)


def _list_overlap(list_q, list_r):
    set_q = set(list_q) if list_q else set()
    set_r = set(list_r) if list_r else set()
    if not set_q or not set_r:
        return None
    intersection = set_q & set_r
    union        = set_q | set_r
    return len(intersection) / len(union)


def _attribute_match(attr, val_q, val_r):
    if ATTRIBUTE_CONFIG[attr] == "list":
        return _list_overlap(val_q, val_r)
    return _scalar_match(val_q, val_r)


def _unique_values(attr, items):
    if ATTRIBUTE_CONFIG[attr] == "list":
        seen = set()
        for item in items:
            for v in (item.get(attr) or []):
                seen.add(v)
        return len(seen)
    else:
        vals = [item.get(attr) for item in items if item.get(attr) is not None]
        return len(set(vals))


def _item_dissimilarity(a, b):
    scores = []
    for attr, atype in ATTRIBUTE_CONFIG.items():
        m = _attribute_match(attr, a.get(attr), b.get(attr))
        if m is not None:
            scores.append(1.0 - m)
    return float(np.mean(scores)) if scores else 0.0


def _intra_list_diversity(items):
    n = len(items)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _item_dissimilarity(items[i], items[j])
            count += 1
    return total / count


def evaluate_profile(embeddings, search_method, aligned_labels, k=10):
    focus_accum   = defaultdict(list)
    variety_accum = defaultdict(list)
    ild_list      = []
    per_query     = []

    for q_idx in range(len(embeddings)):
        query = embeddings[q_idx:q_idx + 1]
        labels, _ = search_method.search(query, k=k + 1)
        labels = np.asarray(labels)
        if labels.ndim > 1:
            labels = labels[0]

        labels = labels[labels != q_idx][:k]

        q_meta  = aligned_labels[q_idx]
        r_metas = [aligned_labels[i] for i in labels]

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
            n_unique = _unique_values(attr, r_metas)
            q_variety[attr] = n_unique
            variety_accum[attr].append(n_unique)

        ild = _intra_list_diversity(r_metas)
        ild_list.append(ild)

        per_query.append({
            "query_idx":  q_idx,
            "query_meta": q_meta,
            "retrieved":  list(labels),
            "focus":      q_focus,
            "variety":    q_variety,
            "ild":        ild,
        })

    report = {
        "focus":   {attr: float(np.mean(v)) if v else None
                    for attr, v in focus_accum.items()},
        "variety": {attr: float(np.mean(v)) if v else None
                    for attr, v in variety_accum.items()},
        "ild":     float(np.mean(ild_list)) if ild_list else None,
        "per_query": per_query,
    }
    return report


def print_profile_report(report, method_name="", backend_name="", k=10):
    title = f"Profile Report — {method_name} / {backend_name}  (k={k})" if method_name else f"Profile Report (k={k})"
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    print("\n📌  FOCUS  (avg match rate vs query, higher = model aligns on this attribute)")
    print(f"  {'Attribute':<14}  {'Match Rate':>12}")
    print(f"  {'-'*30}")
    focus = report["focus"]
    for attr, rate in sorted(focus.items(), key=lambda x: -(x[1] or 0)):
        bar   = _bar(rate, width=20)
        label = f"{rate:.3f}" if rate is not None else "  N/A "
        print(f"  {attr:<14}  {label:>6}  {bar}")

    print("\n🎨  VARIETY  (avg unique values in top-k results, higher = more diverse)")
    print(f"  {'Attribute':<14}  {'Avg Unique':>10}")
    print(f"  {'-'*30}")
    variety = report["variety"]
    for attr, mean_u in sorted(variety.items(), key=lambda x: -(x[1] or 0)):
        label = f"{mean_u:.2f}" if mean_u is not None else "  N/A "
        print(f"  {attr:<14}  {label:>10}")

    print(f"\n🔀  Intra-List Diversity (ILD): {report['ild']:.4f}   (0=identical results, 1=fully varied)")
    print(f"{'='*60}\n")


def _bar(val, width=20):
    if val is None:
        return ""
    filled = int(round(val * width))
    return "█" * filled + "░" * (width - filled)


def compare_profiles(reports: dict, k=10):
    attrs   = list(ATTRIBUTE_CONFIG.keys())
    methods = list(reports.keys())

    print(f"\n{'='*70}")
    print(f"COMPARISON — FOCUS  (avg match rate, k={k})")
    print(f"{'='*70}")
    col = 12
    header = f"  {'Attribute':<14}" + "".join(f"  {m[:col]:<{col}}" for m in methods)
    print(header)
    print("  " + "-" * (14 + (col + 2) * len(methods)))

    for attr in attrs:
        row = f"  {attr:<14}"
        for m in methods:
            val = reports[m]["focus"].get(attr)
            row += f"  {val:.3f}       " if val is not None else f"  {'N/A':<{col}}"
        print(row)

    print(f"\n  {'ILD':<14}" + "".join(
        f"  {reports[m]['ild']:.4f}      " for m in methods))
    print(f"{'='*70}\n")
