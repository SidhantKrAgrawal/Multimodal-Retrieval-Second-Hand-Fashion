#!/usr/bin/env python3
"""
Generate a compact qualitative comparison figure for the paper.
Layout: 4 rows, each a straight horizontal strip of Query + Top-5 results.
  Row 1: DS1  Self-Attention (Tier A)
  Row 2: DS1  Concat Equal   (Tier C)
  Row 3: DS2  Self-Attention (Tier A)
  Row 4: DS2  Concat Equal   (Tier C)

Output: analysis/paper_qual_figure.pdf
"""

import os, json, re
import numpy as np
import hnswlib
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE      = os.path.dirname(os.path.abspath(__file__))
TOP_K  = 5
OUTPUT = os.path.join(BASE, "analysis", "paper_qual_figure.pdf")

# Per-dataset query indices chosen for SA=5/5, CIH=0/5 type-match contrast
# DS1 idx 346:  Ladies / Robe / Red / Winter      — SA=1.00, CIH=0.00
# DS2 idx 1170: Men / T-Shirt / Pink / All season  — SA=1.00, CIH=0.00
QUERY_INDICES = {"Dataset 1": 346, "Dataset 2": 1170}

DATASETS = ["Dataset 1", "Dataset 2"]
METHODS  = ["selfattention", "concat_image_heavy"]

METHOD_LABELS = {
    "selfattention":      "Self-Attention\n(Tier A — Learned)",
    "concat_image_heavy": "Concat\n(Tier C — Heuristic)",
}
DS_LABELS = {
    "Dataset 1": "DS1 · RISE\n(Professional)",
    "Dataset 2": "DS2 · Shpock\n(User-generated)",
}
TIER_A_COLOR = "#1a7a5e"
TIER_C_COLOR = "#888888"
DS1_BORDER   = "#2c5f8a"
DS2_BORDER   = "#8a4f2c"
DS_COLORS    = {"Dataset 1": DS1_BORDER, "Dataset 2": DS2_BORDER}
METHOD_COLORS = {"selfattention": TIER_A_COLOR, "concat_image_heavy": TIER_C_COLOR}

_TS_RE  = re.compile(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})")
_IDX_RE = re.compile(r"^(\d+)$")


# ── helpers ────────────────────────────────────────────────────────────────

def get_item_ids(ds_root):
    dinov2_dir = os.path.join(ds_root, "embeddings", "img_embeddings_dinov2")
    images_dir = os.path.join(ds_root, "images")
    ids = []
    for item_id in sorted(os.listdir(dinov2_dir)):
        stem = os.path.splitext(item_id)[0]
        image_stem = stem[:-4] if stem.endswith("_aug") else stem
        if os.path.exists(os.path.join(images_dir, image_stem + ".jpg")):
            ids.append(item_id)
    return ids


def image_path(ds_root, item_id):
    stem = os.path.splitext(item_id)[0]
    image_stem = stem[:-4] if stem.endswith("_aug") else stem
    return os.path.join(ds_root, "images", image_stem + ".jpg")


def load_label(ds_root, item_id):
    labels_dir = os.path.join(ds_root, "labels")
    stem = os.path.splitext(item_id)[0]
    # DS1: timestamp-based  → labels_TIMESTAMP.json
    m = _TS_RE.search(stem)
    if m:
        path = os.path.join(labels_dir, f"labels_{m.group(1)}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    # DS2: integer-based → STEM.json
    clean = stem[:-4] if stem.endswith("_aug") else stem
    path = os.path.join(labels_dir, f"{clean}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def meta_line(meta):
    parts = [
        meta.get("category", ""),
        meta.get("type", ""),
        (meta.get("colors") or [""])[0] if isinstance(meta.get("colors"), list)
        else meta.get("colors", ""),
        meta.get("season", ""),
    ]
    return " · ".join(p for p in parts if p)


def load_img(path, size=220):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return np.full((size, size, 3), 220, dtype=np.uint8)


def build_index(emb_np):
    n, dim = emb_np.shape
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=n, ef_construction=200, M=16)
    idx.add_items(emb_np, list(range(n)))
    idx.set_ef(50)
    return idx


def retrieve(idx, query_idx, k):
    labels, _ = idx.knn_query(np.array([idx.get_items([query_idx])[0]]), k=k + 1)
    return [l for l in labels[0].tolist() if l != query_idx][:k]


# ── main ───────────────────────────────────────────────────────────────────

def main():
    rows = []

    for ds in DATASETS:
        ds_root   = os.path.join(BASE, ds)
        item_ids  = get_item_ids(ds_root)
        n_items   = len(item_ids)
        query_idx = QUERY_INDICES[ds]
        print(f"\n{ds}: {n_items} items  (query idx {query_idx} → {item_ids[query_idx]})")

        q_id   = item_ids[query_idx]
        q_img  = load_img(image_path(ds_root, q_id))
        q_meta = load_label(ds_root, q_id)
        print(f"  Query: {meta_line(q_meta)}")

        for method in METHODS:
            emb_file = os.path.join(ds_root, "fused_embeddings",
                                    f"fused_item_embeddings_{method}.npy")
            if not os.path.exists(emb_file):
                print(f"  [SKIP] {emb_file} not found")
                continue

            print(f"  Building index for {method}...", end=" ", flush=True)
            emb_np = np.load(emb_file).astype(np.float32)
            idx    = build_index(emb_np)
            print("done")

            result_idxs  = retrieve(idx, query_idx, TOP_K)
            result_imgs  = [load_img(image_path(ds_root, item_ids[i])) for i in result_idxs]
            result_metas = [load_label(ds_root, item_ids[i])            for i in result_idxs]

            rows.append(dict(
                ds=ds, method=method,
                q_img=q_img, q_meta=q_meta,
                result_imgs=result_imgs, result_metas=result_metas,
            ))
            print(f"  ✓ {method}: retrieved {len(result_idxs)} items")

    # ── figure layout: 4 rows × (1 label + 1 query + 5 results) columns ──
    N_ROWS = len(rows)          # 4
    N_COLS = 1 + 1 + TOP_K     # label + query + 5 results = 7

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(20, N_ROWS * 3.5),
        gridspec_kw={"width_ratios": [1.4] + [1] * (N_COLS - 1)},
    )
    fig.patch.set_facecolor("white")

    for row_i, row in enumerate(rows):
        axs         = axes[row_i]
        ds_col      = DS_COLORS[row["ds"]]
        method_col  = METHOD_COLORS[row["method"]]

        # ── col 0: text label ──────────────────────────────────────────────
        ax0 = axs[0]
        ax0.axis("off")
        ax0.set_facecolor("#f7f7f7")
        ax0.text(0.5, 0.62, DS_LABELS[row["ds"]],
                 ha="center", va="center", fontsize=9, fontweight="bold",
                 color=ds_col, transform=ax0.transAxes, linespacing=1.4)
        ax0.text(0.5, 0.28, METHOD_LABELS[row["method"]],
                 ha="center", va="center", fontsize=8,
                 color=method_col, transform=ax0.transAxes, linespacing=1.4)
        # coloured left bar
        ax0.plot([0.06, 0.06], [0.05, 0.95], color=method_col, linewidth=4,
                 transform=ax0.transAxes, clip_on=False)

        # ── col 1: query ───────────────────────────────────────────────────
        ax1 = axs[1]
        ax1.imshow(row["q_img"])
        ax1.set_xticks([]); ax1.set_yticks([])
        for sp in ax1.spines.values():
            sp.set_color(ds_col); sp.set_linewidth(3)
        ax1.set_xlabel(meta_line(row["q_meta"]),
                       fontsize=6.5, labelpad=3, color="#222")
        if row_i == 0:
            ax1.set_title("Query", fontsize=8.5, fontweight="bold", pad=4)
        ax1.text(0.03, 0.95, "Q", transform=ax1.transAxes,
                 fontsize=7, fontweight="bold", color="white", va="top",
                 bbox=dict(boxstyle="round,pad=0.25", fc=ds_col, alpha=0.9))

        # ── cols 2-6: results ──────────────────────────────────────────────
        for res_i in range(TOP_K):
            ax_r = axs[2 + res_i]
            if res_i < len(row["result_imgs"]):
                ax_r.imshow(row["result_imgs"][res_i])
                ax_r.set_xlabel(meta_line(row["result_metas"][res_i]),
                                fontsize=6, labelpad=3, color="#444")
                for sp in ax_r.spines.values():
                    sp.set_color(method_col); sp.set_linewidth(1.5)
            else:
                ax_r.set_facecolor("#eee")
            ax_r.set_xticks([]); ax_r.set_yticks([])
            if row_i == 0:
                ax_r.set_title(f"#{res_i + 1}", fontsize=8, pad=4)

    # separator line between DS1 rows and DS2 rows
    if N_ROWS == 4:
        line_y = axes[1][0].get_position().y0 - 0.005
        fig.add_artist(plt.Line2D(
            [0.07, 0.98], [line_y, line_y],
            transform=fig.transFigure, color="#cccccc", linewidth=1.2, linestyle="--"
        ))

    plt.tight_layout(h_pad=0.6, w_pad=0.4)

    # caption
    q1_meta = rows[0]["q_meta"] if rows else {}
    q2_meta = rows[2]["q_meta"] if len(rows) > 2 else {}
    fig.suptitle(
        f"Figure 2 — Qualitative comparison: Self-Attention (Tier A) vs Concat (Tier C)\n"
        f"DS1 query: {meta_line(q1_meta)}     DS2 query: {meta_line(q2_meta)}",
        fontsize=9, y=1.01, color="#333"
    )

    os.makedirs(os.path.join(BASE, "analysis"), exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        pdf.savefig(fig, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"\n✓ Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
