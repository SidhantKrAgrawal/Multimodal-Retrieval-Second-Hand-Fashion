# Multimodal Item-to-Item Retrieval for Second-Hand Fashion

Code for the paper *Multimodal Item-to-Item Retrieval for Second-Hand Fashion under Noisy Metadata and Weak Visual Consistency*.

## Reproducing the paper's numbers

**Every number in the paper comes from the held-out evaluation:**

```bash
DATASET=1 python -m experiments.heldout_eval
DATASET=2 python -m experiments.heldout_eval
```

`experiments/heldout_eval.py` splits each catalogue into disjoint training (80%) and test (20%) items (seed 42), trains the Self-, Cross- and Co-Attention models on the training items only, and uses every test item as a query against the whole catalogue. It reports all methods (learned fusion, TF-IDF, BM25, Concat Equal / Text-Heavy / Image-Heavy, Late Fusion, Random, single modalities), attribute focus, significance tests and the ground-truth sensitivity analysis. Results are written to `Dataset N/results/heldout_results.json`; the files from our run are included in this repository.

It needs the four embedding folders under `Dataset N/embeddings/` (see *Adding a New Dataset* below for the layout) and the labels under `Dataset N/labels/`.

## Data

**DS1 (RISE, 6,106 items).** Images come from the public *Clothing Dataset for Second-Hand Fashion* (Zenodo, doi:10.5281/zenodo.13788681). Per-item labels are in `Dataset 1/labels/`. Download the images from Zenodo and place them in `Dataset 1/images/`.

**DS2 (Shpock, 4,683 items).** Public second-hand marketplace listings. Attributes extracted with Llama 3.2 are in `Dataset 2/labels/` (`<id>.json`, matching image `<id>.jpg`); postcodes in seller-written text have been replaced with `[POSTCODE]`. Images (about 0.8 GB) are too large for git: download them from https://osf.io/n6s5f/overview?view_only=33018948c2784a718263b2c18209b9ca and unzip to `Dataset 2/images/`.

Embeddings and trained checkpoints are not stored in git. Generate the embeddings with the scripts in `Dataset N/generate_embeddings/`; `experiments/heldout_eval.py` trains the models used in the paper, and `run_pipeline.sh` trains the full-catalogue models.

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv myproject
source myproject/bin/activate          # macOS / Linux
# myproject\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

## Running from Scratch

Run the entire pipeline for a single dataset with one command:

```bash
bash run_pipeline.sh 1        # Dataset 1 (RISE)
bash run_pipeline.sh 2        # Dataset 2 (Shpock)
```

Steps executed in order:

| Step | Script | What it does |
|------|--------|-------------|
| 1–3 | `training/train_*.py` | Train Self-Attention, Cross-Attention, Co-Attention models |
| 4–7 | `fusion/*.py` | Generate fused `.npy` embeddings for all methods |
| 8 | `visualization/visualize_embeddings.py` | UMAP scatter plots per method |
| 9 | `experiments/i2i_runner.py` | Full evaluation: P/R/NDCG + significance tests |
| 10 | `experiments/text_baselines_runner.py` | BM25, TF-IDF, Random baselines |
| 11 | `experiments/i2i_qual_result.py` | Qualitative PDF result sheets |

Each dataset is trained and evaluated independently. There is no cross-dataset transfer.


## Running Individual Scripts

Set the dataset via environment variable (default `"1"`):

```bash
export DATASET=2          # all commands below will use Dataset 2
```

### Train a fusion model

```bash
python -m training.train_selfattention
python -m training.train_crossattention
python -m training.train_coattention
```

Saves weights to `Dataset {N}/models/<model>_trained.pt`.

### Generate fused embeddings

```bash
python -m fusion.self_attention    # requires trained model
python -m fusion.cross_attention   # requires trained model
python -m fusion.co_attention      # requires trained model
python -m fusion.concat            # heuristic — no training needed
```

Saves to `Dataset {N}/fused_embeddings/`.

### Full quantitative evaluation (all fusion methods)

```bash
DATASET=1 python -m experiments.i2i_runner
DATASET=2 python -m experiments.i2i_runner
```

Outputs: P/R/NDCG at k=5,10,20; profile evaluation; significance tests; summary table.
Log saved to `Dataset {N}/dataset{N}_results_i2i_runner.txt`.

### Single-modality baselines

```bash
DATASET=1 python -m experiments.i2i_raw_runner
```

Evaluates each embedding modality (image_dino, image_clip, text_clip, text_qwen) individually.
Log saved to `Dataset 1/dataset1_results_i2i_raw_runner.txt`.

### Ablation study

```bash
DATASET=1 python -m experiments.ablation_runner
```

Tests: Phase 1-only (frozen backbone), no uniformity loss, SVD warm-start, no pseudo-label mining.
Log saved to `Dataset 1/results_ablation_runner.txt`.

### FashionCLIP joint baseline

```bash
DATASET=1 python -m experiments.fashionclip_joint_runner
```

Evaluates the FashionCLIP joint embedding (average of image_clip + text_clip) as a standalone baseline.

### UMAP visualisation

```bash
python visualization/visualize_embeddings.py
```

Saves scatter plots per method to `Dataset {N}/plots/<method>/`.

### Qualitative PDF results

```bash
python -m experiments.i2i_qual_result
```

Saves `Dataset {N}/results/qualitative_results/Query_<idx>/` — one PDF per method per query.


## Text-Retrieval Baselines


```bash
DATASET=1 python -m experiments.text_baselines_runner
DATASET=2 python -m experiments.text_baselines_runner
```


Each item's label dict is converted to a weighted text document:
- `category` ×4, `type` ×3, `colors` ×2, `season`/`brand`/`pattern`/`material` ×1

Also runs significance tests comparing text baselines against embedding-based methods.
Log saved to `Dataset {N}/results_text_baselines_runner.txt`.



## Adding a New Dataset

1. Create the folder structure:

```bash
N=3
mkdir -p "Dataset $N"/{images,labels,models,plots,results,fused_embeddings}
mkdir -p "Dataset $N/embeddings"/{img_embeddings_dinov2,img_embeddings_fashionclip,txt_embeddings_fashionclip,txt_embeddings_qwen}
```

2. Add images to `Dataset N/images/` as `<item_id>.jpg`.

3. Add labels to `Dataset N/labels/` — one JSON file per item named `<item_id>.json`:

```json
{
  "category": "tops",
  "type": "t-shirt",
  "colors": ["white", "blue"],
  "season": "summer",
  "brand": "",
  "pattern": "plain",
  "material": "cotton",
  "condition": "good"
}
```

4. Generate embeddings into the four sub-folders under `Dataset N/embeddings/`. Each `.pt` file should be a 1-D PyTorch tensor saved with `torch.save(tensor, path)`.

5. Run the pipeline:

```bash
bash run_pipeline.sh 3
# or step-by-step:
DATASET=3 python -m training.train_selfattention
DATASET=3 python -m experiments.i2i_runner
DATASET=3 python -m experiments.text_baselines_runner
```
