import os
import math
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.nn.functional as F

from load_embeddings import load_all_embeddings
from evaluation.load_labels import load_all_labels
from evaluation.align_labels import align_labels_with_embeddings
from evaluation.build_ground_truth import item_similarity
from training.hard_negative_sampler import CategoryAwareSampler, PseudoLabelMiner
from training.losses import CombinedLoss
from fusion.concat import concat_fusion, MODALITIES

dataset = os.environ.get("DATASET", "1")  # defaults to "1" if not set

EMBEDDINGS_DIR = f"Dataset {dataset}/embeddings"
LABELS_DIR     = f"Dataset {dataset}/labels"
IMAGES_DIR     = f"Dataset {dataset}/images"


class MultiModalDataset(Dataset):
    def __init__(self, embeddings, labels, item_ids):
        self.embeddings = embeddings
        self.labels     = labels
        self.item_ids   = item_ids
        self.keys       = sorted(["image_clip", "image_dino", "text_clip", "text_qwen"])

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        return {k: self.embeddings[k][idx] for k in self.keys}, idx


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def build_loader(dataset, aligned_labels, batch_size, categories_per_batch):
    sampler = CategoryAwareSampler(
        labels=aligned_labels,
        batch_size=batch_size,
        categories_per_batch=categories_per_batch,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0), sampler


def run_training(model, model_save_path: str, cfg: dict):
    DEVICE               = cfg["DEVICE"]
    BATCH_SIZE           = cfg["BATCH_SIZE"]
    PHASE1_EPOCHS        = cfg["PHASE1_EPOCHS"]
    PHASE2_EPOCHS        = cfg["PHASE2_EPOCHS"]
    LR_PHASE1            = cfg["LR_PHASE1"]
    LR_PHASE2            = cfg["LR_PHASE2"]
    WARMUP_EPOCHS        = cfg["WARMUP_EPOCHS"]
    TEMPERATURE          = cfg["TEMPERATURE"]
    LAMBDA_UNIFORMITY    = cfg["LAMBDA_UNIFORMITY"]
    SIMILARITY_THRESHOLD = cfg["SIMILARITY_THRESHOLD"]
    PSEUDO_COS_THRESHOLD = cfg["PSEUDO_COS_THRESHOLD"]
    SOFT_LABEL_THRESHOLD = cfg["SOFT_LABEL_THRESHOLD"]
    SOFT_WEIGHT          = cfg["SOFT_WEIGHT"]
    CATEGORIES_PER_BATCH = cfg["CATEGORIES_PER_BATCH"]
    MAX_GRAD_NORM        = cfg["MAX_GRAD_NORM"]
    modality_keys        = cfg["modality_keys"]

    print("Loading embeddings and labels...")
    emb         = load_all_embeddings(EMBEDDINGS_DIR)
    labels_dict = load_all_labels(LABELS_DIR)
    item_ids    = emb["item_ids"]

    input_dims = {k: emb[k].shape[1] for k in modality_keys}
    print(f"  Catalogue size : {len(item_ids)} items")
    print(f"  Modality dims  : {input_dims}")
    print(f"  Label entries  : {len(labels_dict)}")

    aligned_labels = align_labels_with_embeddings(item_ids, labels_dict)
    matched = sum(1 for l in aligned_labels if l)
    print(f"  Aligned labels : {matched}/{len(item_ids)} non-empty")

    real_labels = [l for l in aligned_labels if l]
    if len(real_labels) >= 2:
        test_sim = item_similarity(real_labels[0], real_labels[1])
        c0 = f"{real_labels[0].get('category','?')}/{real_labels[0].get('type','?')}"
        c1 = f"{real_labels[1].get('category','?')}/{real_labels[1].get('type','?')}"
        print(f"  item_similarity smoke-test: sim={test_sim:.4f}  ({c0} vs {c1})")
    else:
        print("  WARNING: Not enough real labels to test item_similarity!")

    print("\nBuilding concat embeddings for pseudo-label mining...")
    concat_emb = concat_fusion(emb, MODALITIES)

    dataset = MultiModalDataset(emb, aligned_labels, item_ids)

    miner = PseudoLabelMiner(
        concat_embeddings=concat_emb,
        hard_threshold=SIMILARITY_THRESHOLD,
        pseudo_cos_threshold=PSEUDO_COS_THRESHOLD,
        soft_label_threshold=SOFT_LABEL_THRESHOLD,
        soft_weight=SOFT_WEIGHT,
    )

    if cfg.get("USE_WARMSTART", True):
        model.init_from_concat({k: emb[k] for k in modality_keys})
    else:
        print("Skipping concat warm-start (USE_WARMSTART=False).")

    criterion = CombinedLoss(
        temperature=TEMPERATURE,
        lambda_uniformity=LAMBDA_UNIFORMITY,
    )

    def run_epoch(loader, optimizer, scheduler, epoch_label, diag_batch=False):
        model.train()
        total_loss, total_sc, total_u = 0.0, 0.0, 0.0
        n_batches = 0
        progress  = tqdm(loader, desc=epoch_label)

        for batch_idx, (inputs, indices) in enumerate(progress):
            inputs       = {k: v.to(DEVICE) for k, v in inputs.items()}
            batch_labels = [aligned_labels[int(i)] for i in indices]

            if diag_batch and batch_idx == 0:
                pos_mask, stats = miner.build_pos_mask_with_stats(
                    indices=[int(i) for i in indices],
                    batch_labels=batch_labels,
                    item_similarity_fn=item_similarity,
                    device=DEVICE,
                )
                print(f"\n  [BATCH DIAGNOSTICS — first batch]")
                print(f"    Batch size        : {len(indices)}")
                print(f"    Hard positives    : {stats['hard_positives']}")
                print(f"    Pseudo positives  : {stats['pseudo_positives']}")
                print(f"    Negatives         : {stats['negatives']}")
                print(f"    Positive rate     : {stats['positive_rate']:.4f}")
                print(f"    Mean label sim    : {stats['mean_label_sim']:.4f}")
                print(f"    Max label sim     : {stats['max_label_sim']:.4f}")
                if stats['hard_positives'] == 0 and stats['pseudo_positives'] == 0:
                    print(f"    ⚠ WARNING: ZERO positives in first batch!")
                    print(f"      SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD}")
                    print(f"      Max label sim seen: {stats['max_label_sim']:.4f}")
                    print(f"      If max_label_sim is ~0, label alignment still broken.")
                    print(f"      If max_label_sim > 0 but < threshold, lower the threshold.")
                    print(f"      Sample labels:")
                    for i in range(min(3, len(batch_labels))):
                        print(f"        [{i}]: {dict(list(batch_labels[i].items())[:3]) if batch_labels[i] else '{}'}")
                else:
                    print(f"    ✓ Positives found — training signal is real.")
            else:
                pos_mask = miner.build_pos_mask(
                    indices=[int(i) for i in indices],
                    batch_labels=batch_labels,
                    item_similarity_fn=item_similarity,
                    device=DEVICE,
                )

            optimizer.zero_grad()
            fused = model(inputs)
            loss, components = criterion(fused, pos_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()
            total_sc   += components["supcon"]
            total_u    += components["uniformity"]
            n_batches  += 1
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sc":   f"{components['supcon']:.4f}",
                "u":    f"{components['uniformity']:.4f}",
            })

        if n_batches == 0:
            print(f"  ⚠ No batches produced this epoch!")
            return 0.0, 0.0, 0.0

        return total_loss / n_batches, total_sc / n_batches, total_u / n_batches

    print(f"\n{'='*60}")
    print(f"PHASE 1: Partial freeze ({PHASE1_EPOCHS} epochs, lr={LR_PHASE1})")
    print(f"{'='*60}")

    model.freeze_backbone(frozen=True)

    loader, _ = build_loader(dataset, aligned_labels, BATCH_SIZE, CATEGORIES_PER_BATCH)
    n_batches_estimate = len(loader)

    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1, weight_decay=1e-4,
    )
    total_steps_p1 = PHASE1_EPOCHS * max(n_batches_estimate, 1)
    warmup_steps   = WARMUP_EPOCHS * max(n_batches_estimate, 1)
    scheduler_p1   = get_cosine_schedule_with_warmup(
        optimizer_p1, warmup_steps, total_steps_p1
    )

    for epoch in range(PHASE1_EPOCHS):
        loader, _ = build_loader(dataset, aligned_labels, BATCH_SIZE, CATEGORIES_PER_BATCH)
        lbl = f"P1 Epoch {epoch+1}/{PHASE1_EPOCHS}"
        avg_loss, avg_sc, avg_u = run_epoch(
            loader, optimizer_p1, scheduler_p1, lbl,
            diag_batch=(epoch == 0)
        )
        print(f"{lbl} | loss={avg_loss:.4f} | supcon={avg_sc:.4f} | uniformity={avg_u:.4f}")

    print(f"\n{'='*60}")
    print(f"PHASE 2: Full fine-tune ({PHASE2_EPOCHS} epochs, lr={LR_PHASE2})")
    print(f"{'='*60}")

    model.freeze_backbone(frozen=False)

    loader, _ = build_loader(dataset, aligned_labels, BATCH_SIZE, CATEGORIES_PER_BATCH)
    optimizer_p2 = torch.optim.AdamW(
        model.parameters(), lr=LR_PHASE2, weight_decay=1e-4,
    )
    # warm restarts prevent lr decaying to zero partway through training
    T_0 = max(10, PHASE2_EPOCHS // 5)
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p2, T_0=T_0, T_mult=1, eta_min=1e-6,
    )

    best_loss = float('inf')
    best_state = None

    for epoch in range(PHASE2_EPOCHS):
        loader, _ = build_loader(dataset, aligned_labels, BATCH_SIZE, CATEGORIES_PER_BATCH)
        lbl = f"P2 Epoch {epoch+1}/{PHASE2_EPOCHS}"
        avg_loss, avg_sc, avg_u = run_epoch(
            loader, optimizer_p2, scheduler_p2, lbl,
            diag_batch=(epoch == 0)
        )
        scheduler_p2.step()
        current_lr = optimizer_p2.param_groups[0]['lr']
        print(f"{lbl} | loss={avg_loss:.4f} | supcon={avg_sc:.4f} | uniformity={avg_u:.4f} | lr={current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best checkpoint (loss={best_loss:.4f})")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nSaved model to '{model_save_path}'")
    return model