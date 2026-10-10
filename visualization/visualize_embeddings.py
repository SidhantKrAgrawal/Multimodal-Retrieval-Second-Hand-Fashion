from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA


SCALAR_ATTRS = ["category", "type", "season", "brand", "pattern",
                "price", "condition"]
LIST_ATTRS   = ["colors", "cut"]
ALL_ATTRS    = SCALAR_ATTRS + LIST_ATTRS

_BASE_PALETTE = plt.cm.tab20.colors 


def _get_palette(n: int):
    return [_BASE_PALETTE[i % len(_BASE_PALETTE)] for i in range(n)]


def extract_color_labels(aligned_labels: List[dict], attr: str) -> List[str]:
    result = []
    for m in aligned_labels:
        if attr in LIST_ATTRS:
            vals = m.get(attr) or []
            result.append(str(vals[0]) if vals else "unknown")
        else:
            result.append(str(m.get(attr, "unknown") or "unknown"))
    return result


def _reduce_pca(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    return PCA(n_components=2, random_state=random_state).fit_transform(
        embeddings.astype(np.float32))


def _reduce_tsne(embeddings: np.ndarray, perplexity: float = 30.0,
                 n_iter: int = 1000, random_state: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    n_pre = min(50, embeddings.shape[1])
    x = PCA(n_components=n_pre, random_state=random_state).fit_transform(
        embeddings.astype(np.float32))
    return TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=random_state, init="pca",
                learning_rate="auto").fit_transform(x)


def _reduce_umap(embeddings: np.ndarray, n_neighbors: int = 15,
                 min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    try:
        import umap
    except ImportError:
        raise ImportError("pip install umap-learn")
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                     min_dist=min_dist, random_state=random_state,
                     low_memory=False).fit_transform(
        embeddings.astype(np.float32))


def reduce_embeddings(embeddings: np.ndarray, reducer: str = "umap",
                      **kwargs) -> np.ndarray:
    r = reducer.lower()
    if r == "pca":   return _reduce_pca(embeddings, **kwargs)
    if r == "tsne":  return _reduce_tsne(embeddings, **kwargs)
    if r == "umap":  return _reduce_umap(embeddings, **kwargs)
    raise ValueError(f"Unknown reducer '{reducer}'. Choose: pca | tsne | umap")


def _subsample_indices(n: int, max_samples: int,
                       random_state: int = 42) -> np.ndarray:
    if n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=max_samples, replace=False))


def plot_single_model(
    coords: np.ndarray,
    color_labels: List[str],
    method: str,
    attr: str,
    reducer: str,
    save_path: str,
    figsize: tuple = (7, 6),
    point_size: int = 6,
    alpha: float = 0.55,
    max_legend_items: int = 20,
) -> None:
    from collections import Counter

    unique_labels = sorted(set(color_labels))

    if len(unique_labels) > max_legend_items:
        counts = Counter(color_labels)
        top    = {l for l, _ in counts.most_common(max_legend_items - 1)}
        unique_labels = sorted(top) + ["other"]

    palette   = _get_palette(len(unique_labels))
    color_map = {l: palette[i] for i, l in enumerate(unique_labels)}

    colours = [color_map.get(l, color_map.get("other", (0.7, 0.7, 0.7)))
               for l in color_labels]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(coords[:, 0], coords[:, 1],
               c=colours, s=point_size, alpha=alpha, linewidths=0)

    ax.set_title(f"{method}  |  {reducer.upper()}  |  coloured by {attr}",
                 fontsize=11, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_patches = [mpatches.Patch(color=color_map[l], label=l)
                      for l in unique_labels]
    ax.legend(handles=legend_patches, title=attr,
              loc="best", fontsize=8, title_fontsize=9,
              frameon=True, framealpha=0.7,
              ncol=max(1, len(unique_labels) // 10))

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [viz] -> {save_path}")


def visualize_per_model(
    embeddings_dict: Dict[str, np.ndarray],
    aligned_labels: List[dict],
    attrs: Optional[List[str]] = None,
    reducer: str = "umap",
    max_samples: int = 4000,
    out_dir: str = "plots",
    reducer_kwargs: Optional[dict] = None,
    point_size: int = 6,
    alpha: float = 0.55,
):
    attrs          = attrs or ["category", "type", "colors", "season"]
    reducer_kwargs = reducer_kwargs or {}

    print(f"\n{'='*60}")
    print(f"EMBEDDING VISUALISATION  ({reducer.upper()}, max {max_samples} pts)")
    print(f"  Models : {list(embeddings_dict.keys())}")
    print(f"  Attrs  : {attrs}")
    print(f"  Output : {out_dir}/<model>/")
    print(f"{'='*60}")

    for method, emb in embeddings_dict.items():
        print(f"\n  Model: {method}  ({emb.shape[0]} x {emb.shape[1]})")

        # same subsample indices reused across all attrs to keep plots comparable
        idxs  = _subsample_indices(len(emb), max_samples)
        emb_s = emb[idxs]

        print(f"    Reducing to 2-D via {reducer} ...", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = reduce_embeddings(emb_s, reducer=reducer,
                                       **reducer_kwargs)
        print(f"done  ({len(idxs)} pts)")

        model_dir = os.path.join(out_dir, method)
        os.makedirs(model_dir, exist_ok=True)

        for attr in attrs:
            raw_labels    = extract_color_labels(aligned_labels, attr)
            subset_labels = [raw_labels[i] for i in idxs]
            save_path     = os.path.join(model_dir, f"{reducer}_{attr}.png")

            plot_single_model(
                coords=coords,
                color_labels=subset_labels,
                method=method,
                attr=attr,
                reducer=reducer,
                save_path=save_path,
                point_size=point_size,
                alpha=alpha,
            )

    print(f"\n  [viz] All plots saved under {out_dir}/")


def visualize_query_neighbourhood(
    embeddings: np.ndarray,
    query_idx: int,
    retrieved_indices: List[int],
    aligned_labels: List[dict],
    color_by: str = "category",
    reducer: str = "umap",
    max_context: int = 3000,
    save_path: Optional[str] = None,
    reducer_kwargs: Optional[dict] = None,
):
    reducer_kwargs = reducer_kwargs or {}
    n = len(embeddings)

    highlight_set  = set(retrieved_indices) | {query_idx}
    background_idx = np.array([i for i in range(n) if i not in highlight_set])
    if len(background_idx) > max_context:
        rng = np.random.default_rng(42)
        background_idx = np.sort(rng.choice(background_idx,
                                            size=max_context, replace=False))

    all_idx    = np.concatenate([background_idx,
                                 np.array(retrieved_indices),
                                 np.array([query_idx])])
    emb_subset = embeddings[all_idx]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = reduce_embeddings(emb_subset, reducer=reducer,
                                   **reducer_kwargs)

    n_bg       = len(background_idx)
    n_ret      = len(retrieved_indices)
    coords_bg  = coords[:n_bg]
    coords_ret = coords[n_bg: n_bg + n_ret]
    coords_q   = coords[-1:]

    ret_labels  = extract_color_labels(aligned_labels, color_by)
    ret_lbl_sub = [ret_labels[i] for i in retrieved_indices]
    unique_lbl  = sorted(set(ret_lbl_sub))
    palette     = _get_palette(len(unique_lbl))
    cmap        = {l: palette[i] for i, l in enumerate(unique_lbl)}
    ret_colours = [cmap[l] for l in ret_lbl_sub]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(coords_bg[:, 0], coords_bg[:, 1],
               c=[(0.85, 0.85, 0.85)], s=4, alpha=0.4,
               linewidths=0, zorder=1)
    ax.scatter(coords_ret[:, 0], coords_ret[:, 1],
               c=ret_colours, s=40, alpha=0.9,
               linewidths=0.4, edgecolors="white", zorder=3)
    ax.scatter(coords_q[:, 0], coords_q[:, 1],
               marker="*", c=["gold"], s=300,
               edgecolors="black", linewidths=0.8, zorder=4)
    for rank, (cx, cy) in enumerate(coords_ret):
        ax.annotate(str(rank + 1), (cx, cy),
                    fontsize=6.5, ha="center", va="center",
                    color="white", fontweight="bold", zorder=5)

    legend_patches = [mpatches.Patch(color=cmap[l], label=l)
                      for l in unique_lbl]
    legend_patches.append(mpatches.Patch(color="gold",
                                         label=f"Query [{query_idx}]"))
    ax.legend(handles=legend_patches, title=color_by,
              fontsize=8, title_fontsize=9, frameon=True, loc="best")
    ax.set_title(
        f"Query neighbourhood  |  {reducer.upper()}  |  "
        f"query={query_idx}, k={len(retrieved_indices)}",
        fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Saved -> {save_path}")

    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    dataset              = os.environ.get("DATASET", "1")
    EMBEDDINGS_DIR       = f"Dataset {dataset}/embeddings"
    FUSED_EMBEDDINGS_DIR = f"Dataset {dataset}/fused_embeddings"
    LABELS_DIR           = f"Dataset {dataset}/labels"

    EMBEDDING_FILES = {
        "selfattention":       "fused_item_embeddings_selfattention.npy",
        "coattention":         "fused_item_embeddings_coattention.npy",
        "crossattention":      "fused_item_embeddings_crossattention.npy",
        "concat":              "fused_item_embeddings_concat.npy",
        "concat_text_heavy":   "fused_item_embeddings_concat_text_heavy.npy",
        "concat_image_heavy":  "fused_item_embeddings_concat_image_heavy.npy",
    }

    ATTRS_TO_PLOT  = ["category", "type", "colors", "season"]
    REDUCER        = "umap"
    MAX_SAMPLES    = 4000
    OUT_DIR        = f"Dataset {dataset}/plots"
    REDUCER_KWARGS = {}

    from load_embeddings import load_all_embeddings
    from evaluation.load_labels import load_all_labels
    from evaluation.align_labels import align_labels_with_embeddings

    print("Loading labels ...")
    emb            = load_all_embeddings(EMBEDDINGS_DIR)
    labels_dict    = load_all_labels(LABELS_DIR)
    aligned_labels = align_labels_with_embeddings(emb["item_ids"], labels_dict)

    embeddings_dict = {}
    for method, fname in EMBEDDING_FILES.items():
        path = os.path.join(FUSED_EMBEDDINGS_DIR, fname)
        if os.path.exists(path):
            embeddings_dict[method] = np.load(path)
            print(f"  Loaded {method}: {embeddings_dict[method].shape}")
        else:
            print(f"  [SKIP] {path} not found")

    if not embeddings_dict:
        print("No embedding files found. Check EMBEDDINGS_DIR and EMBEDDING_FILES.")
        sys.exit(1)

    visualize_per_model(
        embeddings_dict=embeddings_dict,
        aligned_labels=aligned_labels,
        attrs=ATTRS_TO_PLOT,
        reducer=REDUCER,
        max_samples=MAX_SAMPLES,
        out_dir=OUT_DIR,
        reducer_kwargs=REDUCER_KWARGS,
    )