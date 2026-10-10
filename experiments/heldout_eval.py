"""
Held-out evaluation (item-disjoint train/test split). Produces every number
reported in the paper.

Run from the repo root, exactly like the other experiments:
    DATASET=1 python -m experiments.heldout_eval
    DATASET=2 python -m experiments.heldout_eval

What it does
  1. Splits the catalogue into disjoint TRAIN (80%) and TEST (20%) items
     (seed 42). Items are grouped by their label key, so no item ends up on
     both sides of the split.
  2. Trains Self-, Cross- and Co-Attention with the SAME code and config as
     training/train_*.py, but only on TRAIN items. Positive/negative pairs are
     therefore built from TRAIN items only. An untrained Self-Attention model
     (same initialisation, no training) is built for the training ablation.
  3. Uses every TEST item as a query, for every method. Each query searches
     the whole catalogue (as a newly listed item would), excluding itself.
     Ground truth is the 0.70 rule between the query and every candidate.
  4. Methods: the three held-out models, the untrained model, TF-IDF, BM25,
     Concat Equal / Text-Heavy / Image-Heavy, Late Fusion, Random, and the
     four single modalities. The models trained on ALL items ("*_seen") are
     also scored on the same queries, for reference only.
  5. Reports, at k=10: P@10, nP@10 (hits / min(10, #relevant)), NDCG@10,
     ILD@10; attribute Focus@10; paired Wilcoxon + Cohen's d; Spearman
     between P@10 and ILD; and ground-truth sensitivity (threshold sweep
     0.50-0.90 and 20 random weight perturbations, as in
     evaluation/gt_sensitivity.py).
  6. Writes Dataset N/results/heldout_results.json, the held-out fused
     embeddings to Dataset N/fused_embeddings/heldout/, and the split to
     Dataset N/results/heldout_split.json (used by generate_heldout_figures.py).

Search is exact cosine for every embedding method.

Optional environment variables
  SEED=42            split + training seed
  TEST_FRAC=0.2      fraction of items held out
  METHODS=selfattention,crossattention,coattention
  P2_EPOCHS=50       (only lower this for a quick smoke test)
  REUSE_MODELS=1     load Dataset N/models/heldout/<name>_heldout.pt if it
                     exists instead of retraining (0 = always retrain)
"""
import os
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon, spearmanr

dataset         = os.environ.get("DATASET", "1")
EMBEDDINGS_DIR  = f"Dataset {dataset}/embeddings"
LABELS_DIR      = f"Dataset {dataset}/labels"
RESULTS_DIR     = f"Dataset {dataset}/results"
MODELS_DIR      = f"Dataset {dataset}/models/heldout"
FULL_MODELS_DIR = f"Dataset {dataset}/models"
FUSED_DIR       = f"Dataset {dataset}/fused_embeddings/heldout"

SEED         = int(os.environ.get("SEED", "42"))
TEST_FRAC    = float(os.environ.get("TEST_FRAC", "0.2"))
METHODS      = os.environ.get("METHODS", "selfattention,crossattention,coattention").split(",")
P2_EPOCHS    = int(os.environ.get("P2_EPOCHS", "50"))
REUSE_MODELS = os.environ.get("REUSE_MODELS", "1") == "1"
K            = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

from load_embeddings import load_all_embeddings
from evaluation.load_labels import load_all_labels
from evaluation.align_labels import align_labels_with_embeddings, _extract_key
from evaluation.metrics import precision_at_k, ndcg_at_k
from evaluation.evaluate_profile import _intra_list_diversity, _attribute_match
from evaluation.gt_sensitivity import _BASE_WEIGHTS
from retrieval.bm25_search import BM25Corpus, TFIDFCorpus
from fusion.concat import (concat_fusion, concat_fusion_weighted, MODALITIES,
                           WEIGHTS_TEXT_HEAVY, WEIGHTS_IMAGE_HEAVY)
import training._train_base as train_base

MODALITY_KEYS = ["image_clip", "image_dino", "text_clip", "text_qwen"]

# Same settings as experiments/i2i_runner.py
LATE_FUSION_WEIGHTS  = {m: 0.8 for m in MODALITY_KEYS}
LATE_FUSION_ALPHA    = 5.0
LATE_FUSION_SEARCH_K = 60

# Methods of Table 1 (P@10/ILD correlation is computed over these)
MAIN_METHODS = ["selfattention", "crossattention", "coattention", "tfidf", "bm25",
                "concat_text_heavy", "late_fusion", "concat_equal",
                "concat_image_heavy", "random"]
# Methods of the original sensitivity analysis (experiments/i2i_runner.py)
SENSITIVITY_METHODS = ["selfattention", "coattention", "crossattention", "concat_equal",
                       "concat_text_heavy", "concat_image_heavy", "late_fusion"]
SENSITIVITY_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
FOCUS_ATTRS = ["category", "type", "colors", "pattern", "season", "brand"]

# Same config as training/train_*.py, except the Phase 2 epoch override.
CFG = dict(
    BATCH_SIZE           = 128,
    PHASE1_EPOCHS        = 5,
    PHASE2_EPOCHS        = P2_EPOCHS,
    LR_PHASE1            = 3e-4,
    LR_PHASE2            = 1e-4,
    WARMUP_EPOCHS        = 2,
    TEMPERATURE          = 0.07,
    LAMBDA_UNIFORMITY    = 0.3,
    SIMILARITY_THRESHOLD = 0.6,
    PSEUDO_COS_THRESHOLD = 0.85,
    SOFT_LABEL_THRESHOLD = 0.3,
    SOFT_WEIGHT          = 0.5,
    CATEGORIES_PER_BATCH = 16,
    MAX_GRAD_NORM        = 1.0,
    DEVICE               = DEVICE,
    modality_keys        = MODALITY_KEYS,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset(emb, idx):
    out = {"item_ids": [emb["item_ids"][i] for i in idx]}
    t = torch.as_tensor(idx, dtype=torch.long)
    for k in MODALITY_KEYS:
        out[k] = emb[k][t]
    return out


def is_aug(item_id):
    return os.path.splitext(item_id)[0].endswith("_aug")


def build_model(name, input_dims):
    if name == "selfattention":
        from fusion.self_attention import MultiModalSelfAttentionFusion
        return MultiModalSelfAttentionFusion(input_dims=input_dims, d_model=1024,
                                             output_dim=1024, nhead=8, num_layers=2,
                                             dropout=0.1)
    if name == "crossattention":
        from fusion.cross_attention import CrossModalFusion
        return CrossModalFusion(input_dims=input_dims, d_model=1024, output_dim=1024,
                                nhead=8, num_layers=2, dropout=0.1)
    if name == "coattention":
        from fusion.co_attention import CoAttentionFusion
        return CoAttentionFusion(input_dims=input_dims, hidden_dim=1024, output_dim=1024,
                                 num_heads=8, dropout=0.1)
    raise ValueError(name)


def train_or_load(name, path, input_dims, cfg):
    set_seed(SEED)
    model = build_model(name, input_dims).to(DEVICE)
    if REUSE_MODELS and os.path.exists(path):
        print(f"\n===== {name}: loading {path} =====")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        return model
    print(f"\n===== Training {name} on TRAIN items only ({path}) =====")
    return train_base.run_training(model, path, cfg)


@torch.no_grad()
def encode(model, emb, batch=512):
    model.eval()
    n = len(emb["item_ids"])
    outs = []
    for s in range(0, n, batch):
        inp = {k: emb[k][s:s + batch].to(DEVICE) for k in MODALITY_KEYS}
        outs.append(model(inp).cpu())
    return F.normalize(torch.cat(outs), dim=1).numpy()


# ── Ranking: every function returns, per query, its top-K candidate indices ──
# `cand` is a (Q, N) boolean mask of allowed candidates for each query.

def top_k(scores, cand):
    scores = np.where(cand, scores, -np.inf)
    part = np.argpartition(-scores, K, axis=1)[:, :K]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    top = np.take_along_axis(part, order, axis=1)
    # drop non-candidates (only happens when fewer than K candidates exist)
    return [[int(i) for i in row if cand[q, i]] for q, row in enumerate(top)]


def rank_embeddings(X, queries, cand):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return top_k(X[queries] @ X.T, cand)


def rank_tfidf(tfidf, queries, cand):
    M = tfidf._matrix
    return top_k((M[queries] @ M.T).toarray(), cand)


def rank_bm25(bm25, queries, cand):
    scores = np.array([bm25._score_query(bm25._docs[q]) for q in queries])
    return top_k(scores, cand)


def rank_random(n, queries, cand, seed=SEED):
    rng = np.random.default_rng(seed)
    return top_k(rng.random((len(queries), n)), cand)


def rank_late_fusion(emb, queries, cand):
    fused = np.zeros(cand.shape)
    for m in MODALITY_KEYS:
        X = F.normalize(emb[m].float(), dim=1).numpy()
        sim = np.where(cand, X[queries] @ X.T, -np.inf)
        # each modality contributes only its own top-SEARCH_K candidates
        top = np.argpartition(-sim, LATE_FUSION_SEARCH_K, axis=1)[:, :LATE_FUSION_SEARCH_K]
        dist = 1.0 - np.take_along_axis(sim, top, axis=1)        # cosine distance
        contrib = LATE_FUSION_WEIGHTS[m] * np.exp(-LATE_FUSION_ALPHA * dist)
        np.put_along_axis(fused, top, np.take_along_axis(fused, top, axis=1) + contrib, axis=1)
    return top_k(np.where(fused > 0, fused, -np.inf), cand & (fused > 0))


# ── Ground truth, vectorised ─────────────────────────────────────────────────
# Reproduces evaluation.build_ground_truth.item_similarity ("main") and
# evaluation.gt_sensitivity._weighted_sim ("perturb") exactly; main() checks
# the vectorised version against item_similarity before using it.

class LabelMatcher:
    def __init__(self, labels):
        self.nonempty = np.array([bool(l) for l in labels])
        self.eq, self.truthy, self.notnone = {}, {}, {}
        for attr in ["category", "type", "season", "brand"]:
            vals = [l.get(attr) if l else None for l in labels]
            ids = {}
            codes = np.array([ids.setdefault(repr(v), len(ids)) for v in vals])
            self.eq[attr] = codes
            self.truthy[attr] = np.array([bool(v) for v in vals])
            self.notnone[attr] = np.array([v is not None for v in vals])
        vocab = {}
        rows = [[vocab.setdefault(c, len(vocab)) for c in (l.get("colors", []) if l else [])]
                for l in labels]
        self.colors = np.zeros((len(labels), max(len(vocab), 1)), dtype=np.float32)
        for i, r in enumerate(rows):
            self.colors[i, r] = 1.0

    def match(self, queries):
        """Per-attribute (Q, N) boolean match matrices."""
        q = np.asarray(queries)
        m = {}
        for attr in ["category", "type", "season", "brand"]:
            m[attr] = self.eq[attr][q][:, None] == self.eq[attr][None, :]
        both = self.nonempty[q][:, None] & self.nonempty[None, :]
        main = {
            "category": m["category"] & self.truthy["category"][q][:, None],
            "type":     m["type"] & self.notnone["type"][q][:, None],
            "colors":   (self.colors[q] @ self.colors.T) > 0,
            "season":   m["season"] & self.truthy["season"][q][:, None],
            "brand":    m["brand"] & self.truthy["brand"][q][:, None],
        }
        perturb = dict(main)
        perturb["season"] = m["season"] & self.notnone["season"][q][:, None]
        perturb["brand"]  = m["brand"] & self.notnone["brand"][q][:, None]
        main = {a: v & both for a, v in main.items()}
        perturb = {a: v & both for a, v in perturb.items()}
        return main, perturb


def similarity(matches, weights):
    return sum(weights[a] * matches[a].astype(np.float64) for a in weights)


def gt_from_sim(sim, cand, threshold):
    rel = (sim >= threshold - 1e-9) & cand
    return [set(np.flatnonzero(r).tolist()) for r in rel]


# ── Scoring ──────────────────────────────────────────────────────────────────

def score(rankings, gt, labels):
    per_q = {"P@10": [], "nP@10": [], "NDCG@10": []}
    ild = []
    for q, retrieved in enumerate(rankings):
        rel = gt[q]
        if len(rel) == 0:
            continue
        hits = len(set(retrieved[:K]) & rel)
        per_q["P@10"].append(precision_at_k(retrieved, rel, K))
        per_q["nP@10"].append(hits / min(K, len(rel)))
        nd = ndcg_at_k(retrieved, rel, K)
        per_q["NDCG@10"].append(nd if nd is not None else 0.0)
        ild.append(_intra_list_diversity([labels[i] for i in retrieved]))
    out = {m: float(np.mean(v)) for m, v in per_q.items()}
    out["ILD@10"] = float(np.mean(ild))
    out["per_query"] = per_q
    return out


def p10_only(rankings, gt):
    return float(np.mean([len(set(r[:K]) & gt[q]) / K
                          for q, r in enumerate(rankings) if gt[q]]))


def attribute_focus(rankings, queries, labels):
    # same per-query averaging as evaluation/evaluate_profile.py
    out = {}
    for attr in FOCUS_ATTRS:
        rates = []
        for q, retrieved in zip(queries, rankings):
            vals = [_attribute_match(attr, labels[q].get(attr), labels[i].get(attr))
                    for i in retrieved]
            vals = [v for v in vals if v is not None]
            if vals:
                rates.append(np.mean(vals))
        out[attr] = float(np.mean(rates)) if rates else None
    return out


def compare(name_a, name_b, res, metric):
    x = np.array(res[name_a]["per_query"][metric])
    y = np.array(res[name_b]["per_query"][metric])
    diff = x - y
    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
    try:
        p = float(wilcoxon(x, y).pvalue)
    except ValueError:
        p = 1.0
    return {"a": name_a, "b": name_b, "metric": metric,
            "delta": float(diff.mean()), "cohens_d": d, "wilcoxon_p": p}


def rank_positions(scores, methods):
    order = sorted(methods, key=lambda m: -scores[m])
    return [order.index(m) for m in methods]


def sensitivity(rankings, matcher, queries, cand, methods):
    main, perturb = matcher.match(queries)

    # threshold sweep (as evaluation.gt_sensitivity.threshold_sweep)
    sweep, positions = {}, []
    for thr in SENSITIVITY_THRESHOLDS:
        gt = gt_from_sim(similarity(main, _BASE_WEIGHTS), cand, thr)
        s = {m: p10_only(rankings[m], gt) for m in methods}
        sweep[thr] = s
        positions.append(rank_positions(s, methods))
    adj = [spearmanr(positions[i], positions[i + 1])[0] for i in range(len(positions) - 1)]

    # weight perturbation (as evaluation.gt_sensitivity.weight_perturbation_analysis)
    rng = np.random.default_rng(42)
    base_gt = gt_from_sim(similarity(main, _BASE_WEIGHTS), cand, 0.7)
    base = rank_positions({m: p10_only(rankings[m], base_gt) for m in methods}, methods)
    rhos = []
    for _ in range(20):
        w = {a: max(0.0, v + rng.normal(0, 0.05)) for a, v in _BASE_WEIGHTS.items()}
        tot = sum(w.values())
        w = {a: v / tot for a, v in w.items()}
        gt = gt_from_sim(similarity(perturb, w), cand, 0.7)
        pos = rank_positions({m: p10_only(rankings[m], gt) for m in methods}, methods)
        rhos.append(spearmanr(base, pos)[0])
    return {
        "threshold_sweep_P@10": {str(t): v for t, v in sweep.items()},
        "threshold_sweep_mean_adjacent_spearman": float(np.nanmean(adj)),
        "threshold_sweep_best_method": [max(s, key=s.get) for s in sweep.values()],
        "perturbation_spearman": [float(r) for r in rhos],
        "perturbation_mean_spearman": float(np.nanmean(rhos)),
    }


def main():
    set_seed(SEED)
    for d in (RESULTS_DIR, MODELS_DIR, FUSED_DIR):
        os.makedirs(d, exist_ok=True)

    print(f"Held-out evaluation | Dataset {dataset} | seed={SEED} | test_frac={TEST_FRAC}")
    emb = load_all_embeddings(EMBEDDINGS_DIR)
    labels_dict = load_all_labels(LABELS_DIR)
    item_ids = emb["item_ids"]
    labels = align_labels_with_embeddings(item_ids, labels_dict)
    N = len(item_ids)

    # ── 1. Item-disjoint split, grouped by label key ────────────────────────
    keys = [_extract_key(i) for i in item_ids]
    uniq = sorted(set(keys))
    rng = random.Random(SEED)
    rng.shuffle(uniq)
    n_test = int(round(len(uniq) * TEST_FRAC))
    test_keys = set(uniq[:n_test])

    train_idx = [i for i, k in enumerate(keys) if k not in test_keys]
    # Skip an _aug file only when its original is also present; in DS1 every
    # file is _aug (embeddings come from the augmented images), so it is the item.
    has_orig  = {k for i, k in enumerate(keys) if not is_aug(item_ids[i])}
    test_idx  = [i for i, k in enumerate(keys)
                 if k in test_keys and not (is_aug(item_ids[i]) and k in has_orig)]
    overlap = set(keys[i] for i in train_idx) & set(keys[i] for i in test_idx)
    assert not overlap, "split leak!"
    print(f"  Train items: {len(train_idx)}  |  Test items (queries): {len(test_idx)}  "
          f"|  overlapping items: {len(overlap)}")
    with open(f"{RESULTS_DIR}/heldout_split.json", "w") as f:
        json.dump({"item_ids": item_ids, "train_idx": train_idx, "test_idx": test_idx}, f)

    # ── 2. Candidates (whole catalogue minus the query) and ground truth ────
    key_arr = np.array(keys)
    cand = key_arr[None, :] != key_arr[test_idx][:, None]

    matcher = LabelMatcher(labels)
    main_match, _ = matcher.match(test_idx)
    sim = similarity(main_match, _BASE_WEIGHTS)
    gt = gt_from_sim(sim, cand, 0.7)
    # check the vectorised similarity against evaluation.build_ground_truth
    from evaluation.build_ground_truth import item_similarity
    check = np.random.default_rng(0)
    for _ in range(20000):
        r, j = int(check.integers(len(test_idx))), int(check.integers(N))
        assert abs(sim[r, j] - item_similarity(labels[test_idx[r]], labels[j])) < 1e-9
    rel = [len(r) for r in gt if r]
    print(f"  Queries with >=1 relevant: {len(rel)}  avg relevant: {np.mean(rel):.1f}  "
          f"queries with <10 relevant: {100 * np.mean(np.array(rel) < K):.1f}%  "
          f"max possible P@10: {np.mean(np.minimum(rel, K)) / K:.3f}")

    # ── 3. Embeddings for every method (all items encoded) ──────────────────
    vectors = {}
    train_emb = subset(emb, train_idx)
    # run_training() loads embeddings itself; point it at the training subset.
    train_base.load_all_embeddings = lambda *a, **kw: train_emb
    input_dims = {k: emb[k].shape[1] for k in MODALITY_KEYS}

    for name in METHODS:
        model = train_or_load(name, f"{MODELS_DIR}/{name}_heldout.pt", input_dims, CFG)
        vectors[name] = encode(model, emb)

        seen_path = f"{FULL_MODELS_DIR}/{name}_trained.pt"
        if os.path.exists(seen_path):
            seen = build_model(name, input_dims).to(DEVICE)
            seen.load_state_dict(torch.load(seen_path, map_location=DEVICE))
            vectors[f"{name}_seen"] = encode(seen, emb)

    # training ablation: same initialisation (incl. warm start), no Phase 2
    untrained = train_or_load("selfattention", f"{MODELS_DIR}/selfattention_untrained.pt",
                              input_dims, {**CFG, "PHASE2_EPOCHS": 0})
    vectors["selfattention_untrained"] = encode(untrained, emb)

    vectors["concat_text_heavy"]  = concat_fusion_weighted(emb, MODALITIES, WEIGHTS_TEXT_HEAVY).numpy()
    vectors["concat_equal"]       = concat_fusion(emb, MODALITIES).numpy()
    vectors["concat_image_heavy"] = concat_fusion_weighted(emb, MODALITIES, WEIGHTS_IMAGE_HEAVY).numpy()
    for m in MODALITY_KEYS:
        vectors[m] = F.normalize(emb[m].float(), dim=1).numpy()

    for name, X in vectors.items():
        np.save(f"{FUSED_DIR}/{name}.npy", X.astype(np.float32))

    # ── 4. Rankings and scores ──────────────────────────────────────────────
    rankings = {m: rank_embeddings(X, test_idx, cand) for m, X in vectors.items()}
    rankings["tfidf"]       = rank_tfidf(TFIDFCorpus(labels), test_idx, cand)
    rankings["bm25"]        = rank_bm25(BM25Corpus(labels, k1=1.5, b=0.75), test_idx, cand)
    rankings["late_fusion"] = rank_late_fusion(emb, test_idx, cand)
    rankings["random"]      = rank_random(N, test_idx, cand)

    res = {m: score(r, gt, labels) for m, r in rankings.items()}
    focus = {m: attribute_focus(r, test_idx, labels) for m, r in rankings.items()}

    pairs = [("selfattention", "tfidf"), ("selfattention", "crossattention"),
             ("selfattention", "coattention"), ("crossattention", "tfidf"),
             ("coattention", "tfidf"), ("tfidf", "concat_text_heavy"),
             ("tfidf", "late_fusion"), ("bm25", "concat_text_heavy"),
             ("selfattention", "concat_text_heavy"), ("selfattention", "concat_equal"),
             ("selfattention", "selfattention_untrained"),
             ("tfidf", "selfattention_untrained"),
             ("selfattention_seen", "selfattention")]
    comps = [compare(a, b, res, metric) for metric in ["P@10", "NDCG@10"]
             for a, b in pairs if a in res and b in res]

    rho_p_ild = spearmanr([res[m]["P@10"] for m in MAIN_METHODS],
                          [res[m]["ILD@10"] for m in MAIN_METHODS])[0]

    print("\nGround-truth sensitivity...")
    sens = sensitivity(rankings, matcher, test_idx, cand, SENSITIVITY_METHODS)

    # ── 5. Report ───────────────────────────────────────────────────────────
    n_q = len(rel)
    print(f"\n{'='*72}\nHELD-OUT RESULTS  (Dataset {dataset}, {n_q} test queries, whole catalogue)\n{'='*72}")
    print(f"{'Method':<26}{'P@10':>8}{'nP@10':>8}{'NDCG@10':>10}{'ILD@10':>9}")
    for m, r in sorted(res.items(), key=lambda x: -x[1]["P@10"]):
        print(f"{m:<26}{r['P@10']:>8.3f}{r['nP@10']:>8.3f}{r['NDCG@10']:>10.3f}{r['ILD@10']:>9.3f}")

    print(f"\nAttribute Focus@10")
    print(f"{'Method':<26}" + "".join(f"{a:>10}" for a in FOCUS_ATTRS))
    for m in MAIN_METHODS:
        print(f"{m:<26}" + "".join(f"{focus[m][a]:>10.3f}" for a in FOCUS_ATTRS))

    print("\nKey comparisons (paired over test queries):")
    for c in comps:
        print(f"  {c['a']} vs {c['b']} [{c['metric']}]: delta={c['delta']:+.3f}  "
              f"d={c['cohens_d']:.3f}  Wilcoxon p={c['wilcoxon_p']:.2e}")

    print(f"\nSpearman(P@10, ILD@10) over Table 1 methods: {rho_p_ild:.3f}")
    print(f"Threshold sweep 0.50-0.90: best method per threshold "
          f"{sens['threshold_sweep_best_method']}, mean adjacent Spearman "
          f"{sens['threshold_sweep_mean_adjacent_spearman']:.4f}")
    print(f"Weight perturbation (20 samples): mean Spearman "
          f"{sens['perturbation_mean_spearman']:.4f}, "
          f"min {min(sens['perturbation_spearman']):.4f}")

    out = {
        "dataset": dataset, "seed": SEED, "test_frac": TEST_FRAC,
        "n_items": N, "n_train_items": len(train_idx), "n_test_items": len(test_idx),
        "n_test_queries": n_q,
        "avg_relevant_per_query": float(np.mean(rel)),
        "pct_queries_lt10_relevant": float(100 * np.mean(np.array(rel) < K)),
        "max_possible_P@10": float(np.mean(np.minimum(rel, K)) / K),
        "results": {m: {k: v for k, v in r.items() if k != "per_query"} for m, r in res.items()},
        "attribute_focus": focus,
        "comparisons": comps,
        "spearman_P10_ILD": float(rho_p_ild),
        "sensitivity": sens,
    }
    path = f"{RESULTS_DIR}/heldout_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
