import torch
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict
from typing import List, Iterator
import torch.nn.functional as F


class CategoryAwareSampler(Sampler):
    """Builds batches with multiple items per category to guarantee positive pairs."""

    def __init__(
        self,
        labels: List[dict],
        batch_size: int = 128,
        categories_per_batch: int = 16,
        category_key: str = "category",
        drop_last: bool = True,
        min_items_per_cat: int = 2,
    ):
        self.batch_size = batch_size
        self.drop_last  = drop_last

        # composite key (category/type) gives 40-60 groups vs 5 for category alone,
        # restoring effective batch size and producing stronger within-type positives
        self.category_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            cat = label.get("category", "") or ""
            typ = label.get("type", "")     or ""
            if isinstance(cat, list):
                cat = cat[0] if cat else ""
            if isinstance(typ, list):
                typ = typ[0] if typ else ""
            cat = str(cat).strip()
            typ = str(typ).strip()

            if cat and typ:
                group = f"{cat}/{typ}"
            elif cat:
                group = cat
            elif typ:
                group = typ
            else:
                group = "unknown"

            self.category_to_indices[group].append(idx)

        items_per_cat_request = max(2, batch_size // categories_per_batch)

        usable = [
            c for c, idxs in self.category_to_indices.items()
            if len(idxs) >= items_per_cat_request
        ]

        if len(usable) < categories_per_batch:
            for trial_items in range(items_per_cat_request - 1, 1, -1):
                trial_usable = [
                    c for c, idxs in self.category_to_indices.items()
                    if len(idxs) >= trial_items
                ]
                trial_cats = min(categories_per_batch, len(trial_usable))
                if trial_cats >= 2 and trial_cats * trial_items <= batch_size:
                    items_per_cat_request = trial_items
                    usable = trial_usable
                    break

        self.items_per_category   = items_per_cat_request
        self.categories_per_batch = min(categories_per_batch, len(usable))
        self._usable_cats         = usable
        self._effective_batch_size = self.categories_per_batch * self.items_per_category

        all_sizes = [len(v) for v in self.category_to_indices.values()]
        print(f"\n  [CategoryAwareSampler]")
        print(f"    Total categories      : {len(self.category_to_indices)}")
        print(f"    Usable categories     : {len(self._usable_cats)}")
        if all_sizes:
            print(f"    Category sizes        : "
                  f"min={min(all_sizes)}, "
                  f"median={int(np.median(all_sizes))}, "
                  f"max={max(all_sizes)}")
        print(f"    categories_per_batch  : {self.categories_per_batch}")
        print(f"    items_per_category    : {self.items_per_category}")
        print(f"    effective_batch_size  : {self._effective_batch_size}")

        if self.categories_per_batch < 2:
            print(f"    ⚠ WARNING: Only {self.categories_per_batch} usable category.")
            print(f"      Falling back to random shuffle sampling.")
            print(f"      Positives may be rare — check your label fields.")
            self._use_fallback = True
        else:
            self._use_fallback = False

        self._num_batches = self._compute_num_batches()
        print(f"    Batches per epoch     : {self._num_batches}")

        if self._num_batches == 0:
            print(f"    ⚠ WARNING: 0 batches will be produced!")
            print(f"      This means training will do nothing.")
            print(f"      Check your labels — are category fields populated?")
            print(f"      Sample label: {labels[0] if labels else 'EMPTY'}")

    def _compute_num_batches(self) -> int:
        if self._use_fallback:
            total = sum(len(v) for v in self.category_to_indices.values())
            return max(1, total // self.batch_size)

        # simulate pointer advancement to get exact count (matches __iter__ logic)
        available_count = {c: len(self.category_to_indices[c])
                           for c in self._usable_cats}
        pointers = {c: 0 for c in self._usable_cats}
        count = 0
        cats  = list(self._usable_cats)

        while True:
            still_usable = [
                c for c in cats
                if pointers[c] + self.items_per_category <= available_count[c]
            ]
            if len(still_usable) < self.categories_per_batch:
                break
            chosen = still_usable[:self.categories_per_batch]
            for c in chosen:
                pointers[c] += self.items_per_category
            count += 1

        return max(count, 0)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        if self._use_fallback:
            yield from self._fallback_iter()
            return

        cats = list(self._usable_cats)
        np.random.shuffle(cats)

        shuffled = {
            c: np.random.permutation(self.category_to_indices[c]).tolist()
            for c in cats
        }
        pointers = {c: 0 for c in cats}
        batches  = []

        while True:
            available = [
                c for c in cats
                if pointers[c] + self.items_per_category <= len(shuffled[c])
            ]
            if len(available) < self.categories_per_batch:
                break

            chosen_cats = np.random.choice(
                available, self.categories_per_batch, replace=False
            )
            batch = []
            for c in chosen_cats:
                p = pointers[c]
                batch.extend(shuffled[c][p: p + self.items_per_category])
                pointers[c] += self.items_per_category

            batches.append(batch)

        np.random.shuffle(batches)
        for b in batches:
            yield b

    def _fallback_iter(self) -> Iterator[List[int]]:
        all_indices = []
        for cat in self.category_to_indices:
            all_indices.extend(self.category_to_indices[cat])
        np.random.shuffle(all_indices)

        for start in range(0, len(all_indices) - self.batch_size + 1,
                           self.batch_size):
            yield all_indices[start: start + self.batch_size]


class PseudoLabelMiner:
    """Mines soft positive pairs using concat embedding cosine similarity as a proxy."""

    def __init__(
        self,
        concat_embeddings: torch.Tensor,    
        hard_threshold: float        = 0.6,
        pseudo_cos_threshold: float  = 0.85,
        soft_label_threshold: float  = 0.3,
        soft_weight: float           = 0.5,
    ):
        self.concat_emb           = F.normalize(concat_embeddings.float(), dim=1).cpu()
        self.hard_threshold       = hard_threshold
        self.pseudo_cos_threshold = pseudo_cos_threshold
        self.soft_label_threshold = soft_label_threshold
        self.soft_weight          = soft_weight
        print(f"  [PseudoLabelMiner] concat_emb: {self.concat_emb.shape}, "
              f"hard_threshold={hard_threshold}, "
              f"pseudo_cos_threshold={pseudo_cos_threshold}")

    def build_pos_mask(
        self,
        indices: List[int],
        batch_labels: List[dict],
        item_similarity_fn,
        device: torch.device,
    ) -> torch.Tensor:
        B = len(indices)
        pos_mask = torch.zeros(B, B, device=device)

        # explicit LongTensor + clamp guards against unexpected index forms from batch_sampler
        idx_tensor = torch.tensor(
            [int(i) for i in indices], dtype=torch.long
        ).clamp(0, self.concat_emb.shape[0] - 1)

        batch_concat = self.concat_emb[idx_tensor].to(device)   # (B, D)
        cos_sim = torch.matmul(batch_concat, batch_concat.T)     # (B, B)

        for i in range(B):
            for j in range(i + 1, B):
                label_sim = item_similarity_fn(batch_labels[i], batch_labels[j])

                if label_sim >= self.hard_threshold:
                    pos_mask[i, j] = 1.0
                    pos_mask[j, i] = 1.0
                elif (cos_sim[i, j].item() >= self.pseudo_cos_threshold
                      and label_sim >= self.soft_label_threshold):
                    pos_mask[i, j] = self.soft_weight
                    pos_mask[j, i] = self.soft_weight

        return pos_mask

    def build_pos_mask_with_stats(
        self,
        indices: List[int],
        batch_labels: List[dict],
        item_similarity_fn,
        device: torch.device,
    ) -> tuple:
        B = len(indices)
        pos_mask = torch.zeros(B, B, device=device)

        idx_tensor = torch.tensor(
            [int(i) for i in indices], dtype=torch.long
        ).clamp(0, self.concat_emb.shape[0] - 1)

        batch_concat = self.concat_emb[idx_tensor].to(device)
        cos_sim = torch.matmul(batch_concat, batch_concat.T)

        pos_count    = 0
        pseudo_count = 0
        neg_count    = 0
        label_sims   = []

        for i in range(B):
            for j in range(i + 1, B):
                label_sim = item_similarity_fn(batch_labels[i], batch_labels[j])
                label_sims.append(label_sim)

                if label_sim >= self.hard_threshold:
                    pos_mask[i, j] = 1.0
                    pos_mask[j, i] = 1.0
                    pos_count += 1
                elif (cos_sim[i, j].item() >= self.pseudo_cos_threshold
                      and label_sim >= self.soft_label_threshold):
                    pos_mask[i, j] = self.soft_weight
                    pos_mask[j, i] = self.soft_weight
                    pseudo_count += 1
                else:
                    neg_count += 1

        total_pairs = B * (B - 1) // 2
        stats = {
            "hard_positives":   pos_count,
            "pseudo_positives": pseudo_count,
            "negatives":        neg_count,
            "total_pairs":      total_pairs,
            "positive_rate":    (pos_count + pseudo_count) / max(total_pairs, 1),
            "mean_label_sim":   float(np.mean(label_sims)) if label_sims else 0.0,
            "max_label_sim":    float(np.max(label_sims))  if label_sims else 0.0,
        }
        return pos_mask, stats