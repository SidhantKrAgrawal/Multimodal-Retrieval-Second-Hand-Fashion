#!/usr/bin/env python3
"""
Paper figures from the held-out setting (run experiments/heldout_eval.py first,
for both datasets). Output: paper_context/ecir_current/heldout_figures/

  umap_selfatt.png, umap_concat.png
      DS1 UMAP of the TEST items only (never seen in training), embedded by
      the held-out Self-Attention model and by Concat Equal; coloured by category.
  qual_ds{1,2}_selfatt.png, qual_ds{1,2}_concat.png
      Query + top-5 for one TEST query per dataset, searching the whole
      catalogue (as in the evaluation).

Query selection rule (fixed, not hand-picked): among TEST items matching
QUERY_FILTER, take the query with the largest difference in top-5 type
matches between Self-Attention and Concat Equal; ties -> lowest index.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_paper_qual_figure import image_path, load_img
from visualization.visualize_embeddings import reduce_embeddings, plot_single_model, extract_color_labels
from evaluation.load_labels import load_all_labels
from evaluation.align_labels import align_labels_with_embeddings, _extract_key

BASE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "paper_context", "ecir_current", "heldout_figures")
TOP_K   = 5

METHODS = {"selfatt": "selfattention", "concat": "concat_equal"}
QUERY_FILTER = {
    "1": lambda l: l.get("category") == "Ladies" and l.get("type") in ("Jacket", "Coat"),
    "2": lambda l: l.get("category") == "Men" and l.get("type") == "T-Shirt",
}


def load(ds):
    root = os.path.join(BASE, f"Dataset {ds}")
    split = json.load(open(os.path.join(root, "results", "heldout_split.json")))
    labels = align_labels_with_embeddings(
        split["item_ids"], load_all_labels(os.path.join(root, "labels")))
    vecs = {m: np.load(os.path.join(root, "fused_embeddings", "heldout", f"{m}.npy"))
            for m in METHODS.values()}
    return root, split, labels, vecs


def top5(X, q, keys):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    s = X @ X[q]
    s[np.array(keys) == keys[q]] = -np.inf          # exclude the query itself
    return list(np.argsort(-s)[:TOP_K])


def caption(l):
    colour = (l.get("colors") or [""])[0]
    return "\n".join(p for p in [l.get("type", ""), colour] if p)


def strip(root, item_ids, labels, q, retrieved, path):
    fig, axes = plt.subplots(1, 1 + TOP_K, figsize=(2.0 * (1 + TOP_K), 2.7))
    for ax, i, title in zip(axes, [q] + retrieved, ["Query"] + [f"#{r + 1}" for r in range(TOP_K)]):
        ax.imshow(load_img(image_path(root, item_ids[i])))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold" if i == q else "normal")
        ax.set_xlabel(caption(labels[i]), fontsize=9)
        match = i == q or labels[i].get("type") == labels[q].get("type")
        for sp in ax.spines.values():
            sp.set_color("black" if i == q else ("#1a7a5e" if match else "#c0392b"))
            sp.set_linewidth(3 if i == q else 2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    chosen = {}

    for ds in ["1", "2"]:
        root, split, labels, vecs = load(ds)
        item_ids, test_idx = split["item_ids"], split["test_idx"]
        keys = [_extract_key(i) for i in item_ids]

        # ── qualitative strips ───────────────────────────────────────────────
        best = None
        for q in test_idx:
            if not QUERY_FILTER[ds](labels[q]):
                continue
            res = {m: top5(vecs[m], q, keys) for m in METHODS.values()}
            hits = {m: sum(labels[i].get("type") == labels[q].get("type") for i in r)
                    for m, r in res.items()}
            gap = hits["selfattention"] - hits["concat_equal"]
            if best is None or gap > best[0]:
                best = (gap, q, res, hits)
        gap, q, res, hits = best
        chosen[ds] = {"query_idx": q, "item_id": item_ids[q], "label": labels[q],
                      "top5_type_matches": hits}
        print(f"DS{ds}: query {item_ids[q]} ({caption(labels[q])!r}) type matches {hits}")
        for short, m in METHODS.items():
            strip(root, item_ids, labels, q, res[m],
                  os.path.join(OUT_DIR, f"qual_ds{ds}_{short}.png"))

        # ── UMAP (DS1, test items only) ──────────────────────────────────────
        if ds == "1":
            test_labels = [labels[i] for i in test_idx]
            colours = extract_color_labels(test_labels, "category")
            for short, m in METHODS.items():
                coords = reduce_embeddings(vecs[m][test_idx], reducer="umap")
                name = "Self-Attention" if m == "selfattention" else "Concat Equal"
                plot_single_model(coords, colours, name, "category", "umap",
                                  os.path.join(OUT_DIR, f"umap_{short}.png"), point_size=10)

    with open(os.path.join(OUT_DIR, "chosen_queries.json"), "w") as f:
        json.dump(chosen, f, indent=2)


if __name__ == "__main__":
    main()
