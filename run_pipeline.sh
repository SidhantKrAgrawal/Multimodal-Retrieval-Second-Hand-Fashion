#!/usr/bin/env bash
# run_pipeline.sh — Run the full pipeline for a selected dataset.
#
# Usage:
#   bash run_pipeline.sh          # interactive: prompts for dataset number
#   bash run_pipeline.sh 1        # non-interactive: run on Dataset 1
#   bash run_pipeline.sh 2        # non-interactive: run on Dataset 2
#
# Each Python script reads the chosen dataset via:
#   import os
#   dataset = os.environ.get("DATASET", "1")

set -euo pipefail

# ─── Dataset selection ────────────────────────────────────────────────────────

if [[ $# -ge 1 ]]; then
    DATASET="$1"
else
    echo "Available datasets:"
    ls -d Dataset\ */ 2>/dev/null | sed 's|/||' | sed 's/^/  /' || echo "  (none found)"
    echo ""
    read -rp "Select dataset number: " DATASET
fi

DATASET_DIR="Dataset ${DATASET}"
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Error: directory '${DATASET_DIR}' not found." >&2
    exit 1
fi

export DATASET
echo ""
echo "Running pipeline on Dataset ${DATASET}"
echo "============================================================"

# ─── Helper ───────────────────────────────────────────────────────────────────

step() {
    local num="$1"
    local desc="$2"
    shift 2
    echo ""
    echo "------------------------------------------------------------"
    echo "Step ${num}: ${desc}"
    echo "------------------------------------------------------------"
    "$@"
}

# ─── Pipeline steps ───────────────────────────────────────────────────────────

# -- Training (learn fusion weights) --
step  1 "Train Co-Attention fusion model"    python -m training.train_coattention
step  2 "Train Cross-Attention fusion model" python -m training.train_crossattention
step  3 "Train Self-Attention fusion model"  python -m training.train_selfattention

# -- Fusion (generate fused embedding .npy files) --
step  4 "Concat fusion (+ PCA variants)"     python -m fusion.concat
step  5 "Co-Attention fusion"                python -m fusion.co_attention
step  6 "Cross-Attention fusion"             python -m fusion.cross_attention
step  7 "Self-Attention fusion"              python -m fusion.self_attention

# -- Visualisation --
step  8 "Embedding visualisation (UMAP)"     python visualization/visualize_embeddings.py

# -- Evaluation --
step  9 "I2I retrieval evaluation"           python -m experiments.i2i_runner

# -- Qualitative results --
step 12 "I2I qualitative PDF results"        python -m experiments.i2i_qual_result

echo ""
echo "============================================================"
echo "Pipeline complete (Dataset ${DATASET})."
echo ""
echo "Outputs written to:"
echo "  Dataset ${DATASET}/models/          — trained fusion model weights"
echo "  Dataset ${DATASET}/fused_embeddings/ — fused .npy embedding files"
echo "  Dataset ${DATASET}/plots/            — UMAP visualisation PNGs"
echo "  Dataset ${DATASET}/results/          — evaluation PDFs"
echo "  Dataset ${DATASET}/results_i2i_runner.txt — full evaluation log"