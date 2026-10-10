import numpy as np
from collections import defaultdict
from scipy import stats

from evaluation.build_ground_truth import build_ground_truth
from evaluation.evaluate_i2i import evaluate_i2i

# base weights used across threshold sweep, perturbation, and leave-one-out
_BASE_WEIGHTS = {
    "category": 0.35,
    "type":     0.25,
    "colors":   0.15,
    "season":   0.10,
    "brand":    0.05,
}


def _weighted_sim(a, b, weights):
    if not a or not b:
        return 0.0
    score = 0.0

    if a.get("category") and a.get("category") == b.get("category"):
        score += weights.get("category", 0.0)

    if a.get("type") is not None and a.get("type") == b.get("type"):
        score += weights.get("type", 0.0)

    ca = set(a.get("colors", []))
    cb = set(b.get("colors", []))
    if ca & cb:
        score += weights.get("colors", 0.0)

    if a.get("season") is not None and a.get("season") == b.get("season"):
        score += weights.get("season", 0.0)

    if a.get("brand") is not None and a.get("brand") == b.get("brand"):
        score += weights.get("brand", 0.0)

    return score


def build_gt_with_weights(labels, threshold, weights):
    gt = defaultdict(set)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i == j:
                continue
            if _weighted_sim(li, lj, weights) >= threshold:
                gt[i].add(j)
    return gt


def build_gt_leave_one_out(labels, threshold, omit_attr):
    weights = dict(_BASE_WEIGHTS)

    if omit_attr not in weights:
        return build_gt_with_weights(labels, threshold, weights)

    weights[omit_attr] = 0.0
    max_possible = sum(weights.values())

    if max_possible <= 0:
        return defaultdict(set)

    # scale threshold proportionally so the bar stays relative to remaining weights
    effective_threshold = threshold * (max_possible / sum(_BASE_WEIGHTS.values()))

    return build_gt_with_weights(labels, effective_threshold, weights)


def _gt_stats(gt, n_items):
    counts = [len(v) for v in gt.values() if v]
    if not counts:
        return 0.0, 0.0, 0
    return (float(np.mean(counts)),
            float(np.median(counts)),
            int(np.max(counts)))


def threshold_sweep(embeddings_dict, search_methods_dict, aligned_labels,
                    thresholds=None, ks=(5, 10, 20)):
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    results = {}
    for thresh in thresholds:
        gt  = build_ground_truth(aligned_labels, threshold=thresh)
        avg, med, mx = _gt_stats(gt, len(aligned_labels))
        row = {"_avg_relevant": avg, "_med_relevant": med, "_max_relevant": mx}

        for method, emb in embeddings_dict.items():
            sm       = search_methods_dict[method]
            row[method] = evaluate_i2i(emb, sm, gt, ks=ks)

        results[thresh] = row

    return results


def print_threshold_sweep(sweep_results, methods, metric="precision", k=10):
    thresholds = sorted(t for t in sweep_results.keys())
    col_w = 11

    print(f"\n{'='*72}")
    print(f"THRESHOLD SWEEP  —  {metric}@{k}")
    print(f"{'='*72}")
    header = f"  {'Threshold':<10}  {'Avg Rel':>8}" + \
             "".join(f"  {m[:col_w]:<{col_w}}" for m in methods)
    print(header)
    print("  " + "-" * (10 + 10 + (col_w + 2) * len(methods)))

    for thresh in thresholds:
        row_data = sweep_results[thresh]
        avg_rel  = row_data.get("_avg_relevant", 0.0)
        row = f"  {thresh:<10.2f}  {avg_rel:>8.1f}"
        for m in methods:
            val = row_data.get(m, {}).get(metric, {}).get(k, None)
            if val is not None:
                row += f"  {val:.4f}     "
            else:
                row += f"  {'N/A':<{col_w}}"
        print(row)

    print()
    _print_rank_stability(sweep_results, methods, metric, k, thresholds)


def _print_rank_stability(sweep_results, methods, metric, k, thresholds):
    rankings = []
    for thresh in thresholds:
        row    = sweep_results[thresh]
        scores = {m: row[m][metric].get(k, 0.0)
                  for m in methods if m in row and metric in row[m]}
        if scores:
            rankings.append(sorted(scores, key=scores.get, reverse=True))

    if not rankings:
        return

    top1s     = [r[0] for r in rankings if r]
    if top1s:
        winner    = max(set(top1s), key=top1s.count)
        win_pct   = top1s.count(winner) / len(top1s) * 100
        print(f"  Rank stability:")
        print(f"    Best method is '{winner}' at "
              f"{win_pct:.0f}% of threshold settings")

    if len(rankings) > 1:
        spearman_vals = []
        for i in range(len(rankings) - 1):
            r1, r2 = rankings[i], rankings[i + 1]
            pos1 = [r1.index(m) if m in r1 else len(r1) for m in methods]
            pos2 = [r2.index(m) if m in r2 else len(r2) for m in methods]
            rho, _ = stats.spearmanr(pos1, pos2)
            if not np.isnan(rho):
                spearman_vals.append(rho)
        if spearman_vals:
            print(f"    Avg Spearman ρ (adjacent thresholds): "
                  f"{np.mean(spearman_vals):.4f}  "
                  f"(1.0 = perfectly stable)")


def weight_perturbation_analysis(embeddings_dict, search_methods_dict,
                                 aligned_labels,
                                 base_threshold=0.7,
                                 n_samples=20,
                                 perturbation_std=0.05,
                                 ks=(5, 10, 20),
                                 metric="precision",
                                 k=10,
                                 seed=42):
    rng = np.random.default_rng(seed)

    gt_base = build_ground_truth(aligned_labels, threshold=base_threshold)
    base_scores = {}
    for method, emb in embeddings_dict.items():
        r = evaluate_i2i(emb, search_methods_dict[method], gt_base, ks=ks)
        base_scores[method] = r[metric][k]

    methods    = list(embeddings_dict.keys())
    base_rank  = sorted(methods, key=base_scores.get, reverse=True)

    spearman_vals     = []
    perturbed_results = []

    for _ in range(n_samples):
        perturbed = {attr: max(0.0, w + rng.normal(0, perturbation_std))
                     for attr, w in _BASE_WEIGHTS.items()}
        total = sum(perturbed.values())
        if total == 0:
            continue
        perturbed = {attr: v / total for attr, v in perturbed.items()}

        gt_p = build_gt_with_weights(aligned_labels, base_threshold, perturbed)

        p_scores = {}
        for method, emb in embeddings_dict.items():
            r = evaluate_i2i(emb, search_methods_dict[method], gt_p, ks=ks)
            p_scores[method] = r[metric][k]

        p_rank = sorted(methods, key=p_scores.get, reverse=True)

        base_pos = [base_rank.index(m) for m in methods]
        p_pos    = [p_rank.index(m)    for m in methods]
        rho, _   = stats.spearmanr(base_pos, p_pos)

        if not np.isnan(rho):
            spearman_vals.append(rho)
            perturbed_results.append((perturbed, p_scores, rho))

    return {
        "base_scores":    base_scores,
        "base_rank":      base_rank,
        "spearman_vals":  spearman_vals,
        "mean_spearman":  float(np.mean(spearman_vals)) if spearman_vals else None,
        "std_spearman":   float(np.std(spearman_vals))  if spearman_vals else None,
        "perturbed":      perturbed_results,
        "metric":         metric,
        "k":              k,
    }


def weight_perturbation_fixed_category(embeddings_dict, search_methods_dict,
                                       aligned_labels,
                                       base_threshold=0.7,
                                       n_samples=20,
                                       perturbation_std=0.05,
                                       ks=(5, 10, 20),
                                       metric="precision",
                                       k=10,
                                       seed=42):
    rng = np.random.default_rng(seed)

    gt_base = build_ground_truth(aligned_labels, threshold=base_threshold)
    base_scores = {}
    for method, emb in embeddings_dict.items():
        r = evaluate_i2i(emb, search_methods_dict[method], gt_base, ks=ks)
        base_scores[method] = r[metric][k]

    methods   = list(embeddings_dict.keys())
    base_rank = sorted(methods, key=base_scores.get, reverse=True)

    FIXED_ATTRS    = {"category"}
    PERTURB_ATTRS  = {a: w for a, w in _BASE_WEIGHTS.items()
                      if a not in FIXED_ATTRS}

    spearman_vals     = []
    perturbed_results = []

    for _ in range(n_samples):
        perturbed = dict(_BASE_WEIGHTS)
        free_perturbed = {
            attr: max(0.0, w + rng.normal(0, perturbation_std))
            for attr, w in PERTURB_ATTRS.items()
        }

        fixed_sum    = sum(_BASE_WEIGHTS[a] for a in FIXED_ATTRS)
        variable_sum = sum(free_perturbed.values())

        if variable_sum <= 0:
            continue

        target_free = 1.0 - fixed_sum
        scale = target_free / variable_sum
        for attr, val in free_perturbed.items():
            perturbed[attr] = val * scale

        gt_p = build_gt_with_weights(aligned_labels, base_threshold, perturbed)

        p_scores = {}
        for method, emb in embeddings_dict.items():
            r = evaluate_i2i(emb, search_methods_dict[method], gt_p, ks=ks)
            p_scores[method] = r[metric][k]

        p_rank = sorted(methods, key=p_scores.get, reverse=True)

        base_pos = [base_rank.index(m) for m in methods]
        p_pos    = [p_rank.index(m)    for m in methods]
        rho, _   = stats.spearmanr(base_pos, p_pos)

        if not np.isnan(rho):
            spearman_vals.append(rho)
            perturbed_results.append((perturbed, p_scores, rho))

    return {
        "base_scores":    base_scores,
        "base_rank":      base_rank,
        "spearman_vals":  spearman_vals,
        "mean_spearman":  float(np.mean(spearman_vals)) if spearman_vals else None,
        "std_spearman":   float(np.std(spearman_vals))  if spearman_vals else None,
        "perturbed":      perturbed_results,
        "metric":         metric,
        "k":              k,
        "fixed_attrs":    list(FIXED_ATTRS),
        "perturbed_attrs": list(PERTURB_ATTRS.keys()),
    }


def print_perturbation_report(result, label=None):
    metric      = result["metric"]
    k           = result["k"]
    fixed_attrs = result.get("fixed_attrs", [])
    free_attrs  = result.get("perturbed_attrs",
                              [a for a in _BASE_WEIGHTS if a not in fixed_attrs])

    if label is None:
        label = ("WEIGHT PERTURBATION (category fixed)"
                 if fixed_attrs else "WEIGHT PERTURBATION (all attributes)")

    print(f"\n{'='*60}")
    print(f"{label}  —  {metric}@{k}")
    print(f"  ({len(result['spearman_vals'])} random weight perturbations)")
    if fixed_attrs:
        print(f"  Fixed  : {', '.join(fixed_attrs)}  "
              f"(weights held at base values)")
        print(f"  Varied : {', '.join(free_attrs)}")
    print(f"{'='*60}")
    print(f"  Baseline ranking:")
    print(f"    {' > '.join(result['base_rank'])}")
    print(f"  Baseline scores:")
    for m, s in sorted(result["base_scores"].items(), key=lambda x: -x[1]):
        print(f"    {m:<28}  {s:.4f}")

    if result["mean_spearman"] is not None:
        print(f"\n  Rank stability across perturbations:")
        print(f"    Mean Spearman ρ : {result['mean_spearman']:.4f}")
        print(f"    Std  Spearman ρ : {result['std_spearman']:.4f}")

        mean_rho = result["mean_spearman"]
        if mean_rho > 0.90:
            verdict = "✅  Highly stable — conclusions robust to weight choice."
        elif mean_rho > 0.70:
            verdict = ("⚠️   Moderately stable — some sensitivity to weights.")
        else:
            verdict = "❌  Unstable — conclusions depend on weight assumptions."
        print(f"\n  Verdict: {verdict}")


def leave_one_out_analysis(embeddings_dict, search_methods_dict,
                           aligned_labels,
                           base_threshold=0.7,
                           ks=(5, 10, 20),
                           metric="precision",
                           k=10):
    attrs = list(_BASE_WEIGHTS.keys())

    gt_base = build_ground_truth(aligned_labels, threshold=base_threshold)
    base    = {method: evaluate_i2i(emb, search_methods_dict[method],
                                    gt_base, ks=ks)[metric][k]
               for method, emb in embeddings_dict.items()}

    loo_results = {"baseline": base, "_base_threshold": base_threshold}

    for omit in attrs:
        # Compute effective threshold for reporting
        remaining = sum(v for a, v in _BASE_WEIGHTS.items() if a != omit)
        effective_thresh = base_threshold * (remaining / sum(_BASE_WEIGHTS.values()))

        gt_loo = build_gt_leave_one_out(aligned_labels, base_threshold, omit)
        avg_rel, _, _ = _gt_stats(gt_loo, len(aligned_labels))

        scores = {method: evaluate_i2i(emb, search_methods_dict[method],
                                        gt_loo, ks=ks)[metric][k]
                  for method, emb in embeddings_dict.items()}

        loo_results[f"omit_{omit}"] = scores
        loo_results[f"omit_{omit}_threshold"] = effective_thresh
        loo_results[f"omit_{omit}_avg_rel"]   = avg_rel

    return loo_results


def print_loo_report(loo_results, methods, metric="precision", k=10):
    attrs  = list(_BASE_WEIGHTS.keys())
    col_w  = 11

    print(f"\n{'='*72}")
    print(f"LEAVE-ONE-ATTRIBUTE-OUT  —  {metric}@{k}")
    print(f"  Δ = score_omitted − baseline_score")
    print(f"  Threshold is scaled proportionally when an attribute is removed.")
    print(f"{'='*72}")

    header = f"  {'Omitted':<10}  {'Eff.Thr':>7}  {'AvgRel':>7}" + \
             "".join(f"  {m[:col_w]:<{col_w}}" for m in methods)
    print(header)
    print("  " + "-" * (10 + 9 + 9 + (col_w + 2) * len(methods)))

    baseline = loo_results["baseline"]
    base_thr = loo_results["_base_threshold"]

    row = f"  {'(baseline)':<10}  {base_thr:>7.3f}  {'':>7}"
    for m in methods:
        row += f"  {baseline.get(m, 0):.4f}     "
    print(row)

    impact = {}
    for omit in attrs:
        key      = f"omit_{omit}"
        scores   = loo_results.get(key, {})
        eff_thr  = loo_results.get(f"{key}_threshold", base_thr)
        avg_rel  = loo_results.get(f"{key}_avg_rel", 0.0)

        row = f"  {omit:<10}  {eff_thr:>7.3f}  {avg_rel:>7.1f}"
        deltas = []
        for m in methods:
            delta  = scores.get(m, 0.0) - baseline.get(m, 0.0)
            deltas.append(abs(delta))
            sign   = "+" if delta >= 0 else ""
            row   += f"  {sign}{delta:.4f}    "
        print(row)
        impact[omit] = float(np.mean(deltas))

    print(f"\n  Attribute impact (avg |Δ| across methods, descending):")
    for attr, imp in sorted(impact.items(), key=lambda x: -x[1]):
        w    = _BASE_WEIGHTS[attr]
        bar  = "█" * int(imp * 100)
        print(f"    {attr:<10}  (w={w:.2f})  avg|Δ|={imp:.4f}  {bar}")


def run_sensitivity_analysis(embeddings_dict, search_methods_dict,
                              aligned_labels,
                              ks=(5, 10, 20),
                              thresholds=None,
                              metric="precision",
                              k=10,
                              run_perturbation=True,
                              n_perturbation_samples=20):
    methods = list(embeddings_dict.keys())

    print("\n" + "█" * 70)
    print("  GROUND TRUTH SENSITIVITY ANALYSIS")
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Reporting: {metric}@{k}")
    print("█" * 70)

    print(f"\n[1/3] Threshold sweep...")
    sweep = threshold_sweep(embeddings_dict, search_methods_dict,
                            aligned_labels, thresholds=thresholds, ks=ks)
    print_threshold_sweep(sweep, methods, metric=metric, k=k)

    if run_perturbation and n_perturbation_samples > 0:
        print(f"\n[2a/3] Weight perturbation — all attributes "
              f"({n_perturbation_samples} samples)...")
        pert_all = weight_perturbation_analysis(
            embeddings_dict, search_methods_dict, aligned_labels,
            n_samples=n_perturbation_samples, ks=ks, metric=metric, k=k)
        print_perturbation_report(
            pert_all,
            label="WEIGHT PERTURBATION (all attributes free)")

        print(f"\n[2b/3] Weight perturbation — category FIXED "
              f"({n_perturbation_samples} samples)...")
        pert_fixed = weight_perturbation_fixed_category(
            embeddings_dict, search_methods_dict, aligned_labels,
            n_samples=n_perturbation_samples, ks=ks, metric=metric, k=k)
        print_perturbation_report(
            pert_fixed,
            label="WEIGHT PERTURBATION (category fixed at 0.35)")

        print(f"\n  ── Perturbation comparison ──────────────────────────────")
        r_all   = pert_all["mean_spearman"]
        r_fixed = pert_fixed["mean_spearman"]
        print(f"    All attributes free  : "
              f"mean ρ = {r_all:.4f}  std = {pert_all['std_spearman']:.4f}")
        print(f"    Category fixed       : "
              f"mean ρ = {r_fixed:.4f}  std = {pert_fixed['std_spearman']:.4f}")
        if r_all is not None and r_fixed is not None:
            gain = r_fixed - r_all
            print(f"    Stability gain from fixing category: Δρ = {gain:+.4f}")
        print(f"  ─────────────────────────────────────────────────────────")
    else:
        print("\n[2/3] Weight perturbation skipped.")

    print(f"\n[3/3] Leave-one-attribute-out...")
    loo = leave_one_out_analysis(
        embeddings_dict, search_methods_dict, aligned_labels,
        ks=ks, metric=metric, k=k)
    print_loo_report(loo, methods, metric=metric, k=k)

    print("\n" + "█" * 70)
    print("  SENSITIVITY ANALYSIS COMPLETE")
    print("█" * 70)