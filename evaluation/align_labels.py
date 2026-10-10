import os
import re

_TIMESTAMP_RE = re.compile(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})")
_INDEX_RE     = re.compile(r"^(\d+)$")


def _extract_key(item_id: str) -> str:
    stem = os.path.splitext(item_id)[0]

    if _INDEX_RE.match(stem):
        return stem

    m = _TIMESTAMP_RE.search(stem)
    if m:
        return m.group(1)

    raise ValueError(
        f"Cannot extract a key from item_id '{item_id}'. "
        f"Expected either a pure integer stem (e.g. '10.pt') "
        f"or a timestamp (e.g. 'front_2022_12_01_09_56_38.pt')."
    )


def align_labels_with_embeddings(item_ids, labels_by_key):
    aligned = []
    missing = []

    for item_id in item_ids:
        key = _extract_key(item_id)

        if key not in labels_by_key:
            missing.append(item_id)
            # empty dict rather than crashing — build_ground_truth handles missing fields
            aligned.append({})
        else:
            aligned.append(labels_by_key[key])

    if missing:
        print(f"[WARNING] align_labels: {len(missing)} item(s) had no matching label "
              f"and were aligned to empty dicts: {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}")

    return aligned