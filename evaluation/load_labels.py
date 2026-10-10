import json
import os
import re

_TIMESTAMP_RE = re.compile(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})")
_INDEX_RE     = re.compile(r"^(\d+)$")


def _detect_scheme(fnames):
    # auto-detect whether labels use integer filenames (dataset2) or timestamps (dataset1)
    stems = [os.path.splitext(f)[0] for f in fnames if f.endswith(".json")]
    if not stems:
        return "index"
    index_count = sum(1 for s in stems if _INDEX_RE.match(s))
    return "index" if index_count / len(stems) > 0.5 else "timestamp"


def load_all_labels(labels_dir):
    fnames = [f for f in os.listdir(labels_dir) if f.endswith(".json")]
    scheme = _detect_scheme(fnames)

    labels = {}
    errors = 0

    for fname in fnames:
        stem = os.path.splitext(fname)[0]

        if scheme == "index":
            if not _INDEX_RE.match(stem):
                continue
            key = stem
        else:
            m = _TIMESTAMP_RE.search(stem)
            if not m:
                continue
            key = m.group(1)

        try:
            with open(os.path.join(labels_dir, fname), "r", encoding="utf-8") as f:
                labels[key] = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load {fname}: {e}")
            errors += 1

    print(f"[INFO] Loaded {len(labels)} labels ({scheme}-based) from '{labels_dir}'"
          + (f"  [{errors} errors]" if errors else ""))
    return labels