import numpy as np
from itertools import combinations
from scipy import stats
from collections import defaultdict

from evaluation.metrics import recall_at_k, ndcg_at_k, precision_at_k


def collect_query_scores(embeddings, search_method, ground_truth,
                         ks=(5, 10, 20)):
    max_k     = max(ks)
    per_query = defaultdict(lambda: defaultdict(list))

    for q in range(len(embeddings)):
        relevant = ground_truth[q]
        if not relevant:
            continue

        query = embeddings[q:q + 1]
        labels, _ = search_method.search(query, k=max_k + 1)
        labels = np.asarray(labels)
        if labels.ndim > 1:
            labels = labels[0]
        retrieved = labels[labels != q]

        for k in ks:
            p = precision_at_k(retrieved, relevant, k)
            r = recall_at_k(retrieved, relevant, k)
            n = ndcg_at_k(retrieved, relevant, k)
            per_query["precision"][k].append(p if p is not None else 0.0)
            per_query["recall"][k].append(r if r is not None else 0.0)
            per_query["ndcg"][k].append(n if n is not None else 0.0)

    return {metric: {k: np.array(vals) for k, vals in kd.items()}
            for metric, kd in per_query.items()}


def collect_query_scores_late_fusion(embeddings, indices, weights,
                                     ground_truth, ks=(5, 10, 20),
                                     fusion_fn=None, alpha=5.0,
                                     search_k=None):
    from retrieval.late_fusion import weighted_score_fusion
    if fusion_fn is None:
        fusion_fn = weighted_score_fusion
    if search_k is None:
        search_k = max(ks) * 3

    max_k     = max(ks)
    n         = len(next(iter(embeddings.values())))
    per_query = defaultdict(lambda: defaultdict(list))

    for q in range(n):
        relevant = ground_truth[q]
        if not relevant:
            continue

        results_buffer = {}
        for m, idx in indices.items():
            q_vec = embeddings[m][q:q + 1]
            labels, distances = idx.search(q_vec, k=search_k + 1)
            labels    = np.asarray(labels[0])
            distances = np.asarray(distances[0])
            mask = labels != q
            results_buffer[m] = (labels[mask], distances[mask])

        fused     = fusion_fn(results_buffer, weights, alpha=alpha)
        retrieved = np.array([i for i, _ in fused[:max_k + 1]])

        for k in ks:
            p = precision_at_k(retrieved, relevant, k)
            r = recall_at_k(retrieved, relevant, k)
            n_val = ndcg_at_k(retrieved, relevant, k)
            per_query["precision"][k].append(p if p is not None else 0.0)
            per_query["recall"][k].append(r if r is not None else 0.0)
            per_query["ndcg"][k].append(n_val if n_val is not None else 0.0)

    return {metric: {k: np.array(vals) for k, vals in kd.items()}
            for metric, kd in per_query.items()}


def _cohens_d(a, b):
    diff = a - b
    sd   = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def _magnitude_label(d):
    d = abs(d)
    if d >= 0.8:  return "large"
    if d >= 0.5:  return "medium"
    if d >= 0.2:  return "small"
    return "negligible"


def compare_two_methods(scores_a, scores_b, ks=(5, 10, 20), alpha=0.05):
    results = {}

    for metric in ["precision", "recall", "ndcg"]:
        for k in sorted(ks):
            a = scores_a[metric][k]
            b = scores_b[metric][k]

            min_len = min(len(a), len(b))
            if min_len == 0:
                continue
            a, b = a[:min_len], b[:min_len]

            mean_a = float(np.mean(a))
            mean_b = float(np.mean(b))
            delta  = mean_a - mean_b

            try:
                w_stat, w_pval = stats.wilcoxon(
                    a, b, zero_method="wilcox", alternative="two-sided")
                w_sig = bool(w_pval < alpha)
            except Exception:
                w_stat = w_pval = None
                w_sig  = False

            try:
                t_stat, t_pval = stats.ttest_rel(a, b)
                t_sig = bool(t_pval < alpha)
            except Exception:
                t_stat = t_pval = None
                t_sig  = False

            cd  = _cohens_d(a, b)
            mag = _magnitude_label(cd)

            results[(metric, k)] = {
                "mean_a":        mean_a,
                "mean_b":        mean_b,
                "delta":         delta,
                "n_queries":     min_len,
                "wilcoxon_stat": w_stat,
                "wilcoxon_pval": w_pval,
                "wilcoxon_sig":  w_sig,
                "ttest_stat":    t_stat,
                "ttest_pval":    t_pval,
                "ttest_sig":     t_sig,
                "cohens_d":      cd,
                "magnitude":     mag,
                "significant":   w_sig,   # primary verdict
            }

    return results


def run_significance_tests(all_scores, ks=(5, 10, 20), alpha=0.05,
                            baseline=None):
    methods = list(all_scores.keys())

    if baseline:
        if baseline not in all_scores:
            raise ValueError(
                f"Baseline '{baseline}' not found in scores. "
                f"Available: {methods}")
        pairs = [(baseline, m) for m in methods if m != baseline]
    else:
        pairs = list(combinations(methods, 2))

    return {(a, b): compare_two_methods(all_scores[a], all_scores[b],
                                         ks=ks, alpha=alpha)
            for a, b in pairs}


def print_significance_report(report, alpha=0.05):
    SIG   = "✅  sig."
    NOSIG = "❌  n.s."

    for (method_a, method_b), results in report.items():
        print(f"\n{'='*72}")
        print(f"  {method_a}  vs  {method_b}")
        print(f"  α={alpha} | primary: Wilcoxon signed-rank | secondary: paired t-test")
        print(f"{'='*72}")
        print(f"  {'Metric':<14} {'Δ mean':>10}  {'W p-val':>9}  "
              f"{'t p-val':>9}  {'d':>6}  {'size':>10}  Verdict")
        print(f"  {'-'*70}")

        for metric in ["precision", "recall", "ndcg"]:
            for k in sorted({kk for _, kk in results}):
                key = (metric, k)
                if key not in results:
                    continue
                r   = results[key]
                d   = r["delta"]
                wp  = r["wilcoxon_pval"]
                tp  = r["ttest_pval"]
                cd  = r["cohens_d"]
                mag = r["magnitude"]
                sig = r["significant"]

                dir_str = f"+{d:.4f} ▲" if d > 0 else (
                          f"{d:.4f} ▼" if d < 0 else f" {d:.4f}  ")
                wp_str  = f"{wp:.4f}" if wp is not None else "   N/A"
                tp_str  = f"{tp:.4f}" if tp is not None else "   N/A"
                cd_str  = f"{cd:+.3f}" if cd is not None else "   N/A"
                verdict = SIG if sig else NOSIG

                print(f"  {metric}@{k:<9} {dir_str:>10}  {wp_str:>9}  "
                      f"{tp_str:>9}  {cd_str:>6}  {mag:>10}  {verdict}")

        n = next(iter(results.values()), {}).get("n_queries", "?")
        print(f"\n  N queries: {n}")


def print_significance_summary_table(report, ks=(5, 10, 20), alpha=0.05):
    metrics_ks  = [(m, k) for m in ["precision", "recall", "ndcg"]
                   for k in sorted(ks)]
    header_cols = [f"{m[0].upper()}@{k}" for m, k in metrics_ks]
    col_w       = 11

    sep = "=" * (36 + col_w * len(header_cols))
    print(f"\n{sep}")
    print("SIGNIFICANCE SUMMARY TABLE")
    print("Δ = method_a − method_b  |  * p<0.05  ** p<0.01  *** p<0.001")
    print("Primary test: Wilcoxon signed-rank (two-sided)")
    print(sep)

    header = f"  {'Pair':<34}" + "".join(f"{h:>{col_w}}" for h in header_cols)
    print(header)
    print("  " + "-" * (34 + col_w * len(header_cols)))

    for (method_a, method_b), results in report.items():
        pair_label = f"{method_a[:16]} vs {method_b[:16]}"
        row = f"  {pair_label:<34}"
        for metric, k in metrics_ks:
            r = results.get((metric, k))
            if r is None:
                row += f"{'N/A':>{col_w}}"
                continue
            p = r["wilcoxon_pval"]
            d = r["delta"]
            if p is None:
                row += f"{'N/A':>{col_w}}"
                continue
            stars = ("***" if p < 0.001 else
                     "**"  if p < 0.01  else
                     "*"   if p < 0.05  else "")
            sign  = "+" if d > 0 else ""
            cell  = f"{sign}{d:.3f}{stars}"
            row  += f"{cell:>{col_w}}"
        print(row)

    print(f"\n  Δ = method_a − method_b.  Positive = method_a better.")
    print(f"  * p<0.05  ** p<0.01  *** p<0.001  (Wilcoxon signed-rank, two-sided)")
    print(f"  Effect sizes (Cohen's d) available in verbose report.")
    print(sep)


def print_effect_size_summary(report, k=10):
    print(f"\n{'='*60}")
    print(f"EFFECT SIZE SUMMARY  (Cohen's d, k={k})")
    print(f"Bands: negligible <0.2 | small 0.2 | medium 0.5 | large 0.8")
    print(f"{'='*60}")
    print(f"  {'Pair':<34}  {'P@k':>8}  {'R@k':>8}  {'N@k':>8}  Size")
    print(f"  {'-'*58}")

    for (method_a, method_b), results in report.items():
        label = f"{method_a[:15]} vs {method_b[:15]}"
        vals  = []
        for metric in ["precision", "recall", "ndcg"]:
            r = results.get((metric, k))
            vals.append(r["cohens_d"] if r else None)

        p_d, r_d, n_d = vals
        avg_d = np.mean([abs(v) for v in vals if v is not None])
        mag   = _magnitude_label(avg_d)

        p_str = f"{p_d:+.3f}" if p_d is not None else "  N/A"
        r_str = f"{r_d:+.3f}" if r_d is not None else "  N/A"
        n_str = f"{n_d:+.3f}" if n_d is not None else "  N/A"

        print(f"  {label:<34}  {p_str:>8}  {r_str:>8}  {n_str:>8}  {mag}")