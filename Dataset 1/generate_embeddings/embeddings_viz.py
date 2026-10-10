import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import mplcursors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from umap import UMAP
from matplotlib.patches import Patch

image_dir = Path("../dataset1/augmented_images")
embedding_dir = Path("../dataset1/img_embeddings_dinov2")
captions_file = Path("captions_updated.txt")

captions = {}
if captions_file.exists():
    with open(captions_file, "r") as f:
        for line in f:
            if ":" in line:
                key, cap = line.strip().split(":", 1)
                captions[key.strip()] = cap.strip()
else:
    print("Captions file not found. No captions will be displayed.")

image_filenames = list(embedding_dir.glob("*.pt"))
embeddings = {file.stem: torch.load(file) for file in image_filenames}
print(f"Loaded {len(embeddings)} embeddings.")

image_names = list(embeddings.keys())
embedding_matrix = torch.cat([embeddings[name] for name in image_names], dim=0)

def on_scroll(event):
    ax = event.inaxes
    if ax is None:
        return
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    xdata = event.xdata
    ydata = event.ydata
    scale_factor = 1/1.5 if event.button == 'up' else 1.5 if event.button == 'down' else 1
    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
    relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
    ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
    ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
    plt.draw()

def visualize_kmeans_umap(n_neighbors=100, min_dist=0.1, n_clusters=10):
    X = embedding_matrix.cpu().numpy()
    umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_embedded = umap_model.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=8)
    kmeans_labels = kmeans.fit_predict(X_embedded)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1],
                         c=kmeans_labels, cmap='tab10', alpha=0.7)

    predefined_labels = {
        0: "T-Shirts, Tank Tops, Skirt",
        1: "Ladies Trousers",
        2: "Cardigans",
        3: "Ladies Floral, Geometric and Animal Print Blouses and Tops",
        4: "Hoodies and Sweaters",
        5: "Ladies Dresses, Striped, Garments Pictured from Afar",
        6: "Denim Jeans",
        7: "Mens Shirts, Blazers",
        8: "Outerwear",
        9: "Shorts, Childrens Pajamas "
    }
    cmap = plt.get_cmap('tab10')
    legend_elements = [
        Patch(facecolor=cmap(i), edgecolor='black', label=label)
        for i, label in predefined_labels.items()
    ]
    ax.legend(handles=legend_elements, title="Cluster Labels", loc='lower right')

    ax.set_title("DINOv2 — UMAP + KMeans Clustering")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    plt.savefig("output_kmeans_umap_with_legend.jpg", format="jpg", dpi=300)

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    current_annotation = [None, None]

    cursor = mplcursors.cursor(scatter, hover=False)

    @cursor.connect("add")
    def on_add(sel):
        if current_annotation[0] is not None:
            current_annotation[0].remove()
            current_annotation[1].remove()
            current_annotation[0] = None
            current_annotation[1] = None
        
        sel.annotation.set_visible(False)
        index = sel.index
        img_filename = image_names[index] + ".jpg"
        clean_name = image_names[index].removeprefix("front_").removesuffix("_aug")
        print("Clicked on:", img_filename)
        img_path = image_dir / img_filename

        try:
            img = plt.imread(str(img_path))
        except Exception as e:
            img = None

        caption_text = captions.get(clean_name, "No caption available")

        if img is not None:
            oi = OffsetImage(img, zoom=0.3, alpha=0.7)
            ab = AnnotationBbox(oi, sel.target, xycoords='data',
                                boxcoords="offset points", pad=0.3, frameon=False)
            ax.add_artist(ab)
            text_ab = ax.text(sel.target[0], sel.target[1],
                              caption_text,
                              fontsize=10, color="black",
                              bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))
            current_annotation[0] = ab
            current_annotation[1] = text_ab
        else:
            sel.annotation.set_text("Image not found")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Loaded", len(embeddings), "embeddings.")
    visualize_kmeans_umap()
