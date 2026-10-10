"""
make2.py
=========
Converts labels.csv → per-item JSON files in the labels/ folder,
using Ollama (llama3.2, local, free) to extract rich structured attributes
from the raw title + description instead of relying on regex.

Output schema matches Dataset 1/labels/*.json exactly:
  brand, brandtext, category, type, size, colors, season,
  pilling (1-5 int), condition (1-5 int), price (range string),
  annotator, cut, pattern, trend, smell, stains, holes, damage,
  material, comment, usage

Checkpointing: already-written JSON files are skipped so the script
can be safely interrupted and re-run.

Usage:
    pip install ollama pandas
    ollama pull llama3.2   (if not already done)
    python make2.py
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import ollama

# ── Config ─────────────────────────────────────────────────────────────────
CSV_PATH       = "labels.csv"
OUTPUT_FOLDER  = "labels"
MODEL          = "llama3.2"
MAX_WORKERS    = 4    # parallel threads (increase if your machine handles it)
# ───────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SYSTEM_PROMPT = """You are a fashion data extraction assistant for a second-hand clothing dataset.
Given a listing title, description, and any known metadata, extract structured attributes.
Return ONLY a valid JSON object — no markdown, no code fences, no extra text whatsoever.

JSON schema (ALL fields required):
{
  "brand":      string,
  "brandtext":  string,
  "category":   string,
  "type":       string,
  "size":       string,
  "colors":     [string],
  "season":     string,
  "pilling":    int,
  "condition":  int,
  "price":      string,
  "annotator":  string,
  "cut":        [string],
  "pattern":    string,
  "trend":      string,
  "smell":      string,
  "stains":     string,
  "holes":      string,
  "damage":     string,
  "material":   string,
  "comment":    string,
  "usage":      string
}

Rules:
- brand: known brand name, "Not in the list" if brand visible but unknown, or ""
- brandtext: brand text in description that differs from brand field, or ""
- category: "Men", "Women", "Children", "Unisex", or "" ("Mens"→"Men", "Ladies"→"Women")
- type: garment type e.g. "Hoodie", "Sweatshirt", "Jacket", "T-Shirt", "Jeans", "Trousers"
- size: e.g. "M", "L", "XL", "10", or ""
- colors: list e.g. ["Black"] or ["Blue", "Grey"]
- season: ALWAYS infer from garment type — never leave blank unless completely unclassifiable.
  Use these mappings:
  Winter → Heavy coat, Puffer jacket, Parka, Ski jacket, Fleece, Thermal, Wool jumper
  Autumn → Hoodie, Sweatshirt, Light jacket, Cardigan, Blazer, Denim jacket, Trench coat
  Summer → T-Shirt, Tank top, Shorts, Dress (light), Swimwear, Polo shirt, Linen shirt
  Spring → Light blouse, Chinos, Casual shirt, Light dress, Raincoat
  All season → Jeans, Trousers, Smart shirt, Leggings, Activewear, Suit, Underwear
  Use "Autumn" for hoodies and sweatshirts. Use "Winter" for thick wool/knit jumpers.
  Allowed values: "Winter", "Summer", "Autumn", "Spring", "All season"
- pilling: 1–5 integer (1=heavy pilling, 5=no pilling; mirror condition if unknown)
- condition: 1–5 integer (1=very poor, 2=poor, 3=fair, 4=good, 5=like new/new)
- price: range string if mentioned: "<20","20-50","50-100","100-150","150-200","200-300","300-400",">400", or ""
- annotator: always "ollama-llama3.2"
- cut: e.g. ["Regular"], ["Slim Fit"], ["Oversized"], or []
- pattern: "None", "Striped", "Checked", "Floral", "Graphic", "Solid", or ""
- trend: "None", "Streetwear", "Vintage", "Smart Casual", "Athleisure", or "None"
- smell: "None", "Minor", or "Major"
- stains: "None", "Minor", or "Major"
- holes: "None", "Minor", or "Major"
- damage: other damage description, or ""
- material: e.g. "100%cotton", "50%wool 50%polyester", or ""
- comment: any other notable detail, or ""
- usage: "Reuse", "Export", "Recycle", "Repair", or ""

If no description, infer from title only. Default smell/stains/holes to "None" unless explicitly mentioned."""


def _build_user_prompt(row: dict) -> str:
    parts = []
    title = str(row.get("Title", "") or "").strip()
    desc  = str(row.get("Description", "") or "").strip()
    if title and title.lower() not in {"n/a", "nan", "none", "just a moment..."}:
        parts.append(f"Title: {title}")
    if desc and desc.lower() not in {"n/a", "nan", "none"}:
        parts.append(f"Description: {desc}")

    hints = {}
    for csv_col, label in [
        ("Gender",    "known_category"),
        ("Style",     "known_type"),
        ("Brand",     "known_brand"),
        ("Colour",    "known_colours"),
        ("Size",      "known_size"),
        ("Condition", "known_condition"),
    ]:
        val = str(row.get(csv_col, "") or "").strip()
        # CSV values often have a leading ": " prefix — strip it
        val = re.sub(r"^:\s*", "", val).strip()
        if val and val.lower() not in {"n/a", "nan", "none", ""}:
            hints[label] = val
    if hints:
        parts.append("Known CSV metadata (use as hints, may be incomplete): " + json.dumps(hints))

    return "\n".join(parts) if parts else "No listing data provided."


def process_row(idx: int, row: dict) -> tuple[int, dict | None]:
    out_path = os.path.join(OUTPUT_FOLDER, f"{idx}.json")
    if os.path.exists(out_path):
        return idx, None  # checkpoint — already done

    user_prompt = _build_user_prompt(row)

    for attempt in range(3):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                options={"temperature": 0.1},
            )
            raw = response["message"]["content"].strip()
            # Strip accidental markdown fences
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$",     "", raw)
            # Extract first {...} block in case of extra text
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group(0)
            data = json.loads(raw)

            # Enforce types and defaults
            data["annotator"] = "ollama-llama3.2"
            data["pilling"]   = int(data.get("pilling", 3))
            data["condition"] = int(data.get("condition", 3))
            for field in ["brand","brandtext","category","type","size","season",
                          "price","pattern","trend","smell","stains","holes",
                          "damage","material","comment","usage"]:
                data.setdefault(field, "")
            for field in ["colors", "cut"]:
                if not isinstance(data.get(field), list):
                    data[field] = []

            return idx, data

        except Exception as exc:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f"  [WARN] row {idx} failed after 3 attempts: {exc}")
                return idx, None

    return idx, None


def main():
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    print(f"Loaded {total} rows from {CSV_PATH}")

    already_done = sum(
        1 for i in range(total)
        if os.path.exists(os.path.join(OUTPUT_FOLDER, f"{i}.json"))
    )
    if already_done:
        print(f"  Resuming — skipping {already_done} already-processed rows")

    rows = [(idx, row.to_dict()) for idx, row in df.iterrows()]

    written = 0
    failed  = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, idx, row): idx for idx, row in rows}
        for future in as_completed(futures):
            idx, data = future.result()
            if data is None:
                if not os.path.exists(os.path.join(OUTPUT_FOLDER, f"{idx}.json")):
                    failed += 1
                continue
            out_path = os.path.join(OUTPUT_FOLDER, f"{idx}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            written += 1
            if written % 100 == 0:
                done_pct = 100 * (already_done + written) / total
                print(f"  Progress: {already_done + written}/{total} ({done_pct:.1f}%)")

    print(f"\nDone.  Written: {written}  |  Failed: {failed}  |  Skipped (checkpoint): {already_done}")


if __name__ == "__main__":
    main()

'''
Loaded 4742 rows from labels.csv
  Progress: 100/4742 (2.1%)
  Progress: 200/4742 (4.2%)
  Progress: 300/4742 (6.3%)
  Progress: 400/4742 (8.4%)
  Progress: 500/4742 (10.5%)
  Progress: 600/4742 (12.7%)
  Progress: 700/4742 (14.8%)
  Progress: 800/4742 (16.9%)
  Progress: 900/4742 (19.0%)
  Progress: 1000/4742 (21.1%)
  Progress: 1100/4742 (23.2%)
  Progress: 1200/4742 (25.3%)
  Progress: 1300/4742 (27.4%)
  [WARN] row 1308 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  [WARN] row 1329 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  [WARN] row 1330 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  [WARN] row 1336 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  [WARN] row 1338 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  [WARN] row 1339 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  [WARN] row 1341 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  Progress: 1400/4742 (29.5%)
  Progress: 1500/4742 (31.6%)
  Progress: 1600/4742 (33.7%)
  Progress: 1700/4742 (35.8%)
  Progress: 1800/4742 (38.0%)
  Progress: 1900/4742 (40.1%)
  [WARN] row 1940 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  Progress: 2000/4742 (42.2%)
  Progress: 2100/4742 (44.3%)
  Progress: 2200/4742 (46.4%)
  Progress: 2300/4742 (48.5%)
  Progress: 2400/4742 (50.6%)
  Progress: 2500/4742 (52.7%)
  Progress: 2600/4742 (54.8%)
  Progress: 2700/4742 (56.9%)
  Progress: 2800/4742 (59.0%)
  [WARN] row 2881 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  Progress: 2900/4742 (61.2%)
  Progress: 3000/4742 (63.3%)
  [WARN] row 3036 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  [WARN] row 3071 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  Progress: 3100/4742 (65.4%)
  [WARN] row 3154 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  Progress: 3200/4742 (67.5%)
  [WARN] row 3280 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  Progress: 3300/4742 (69.6%)
  Progress: 3400/4742 (71.7%)
  Progress: 3500/4742 (73.8%)
  Progress: 3600/4742 (75.9%)
  Progress: 3700/4742 (78.0%)
  [WARN] row 3710 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  Progress: 3800/4742 (80.1%)
  Progress: 3900/4742 (82.2%)
  Progress: 4000/4742 (84.4%)
  Progress: 4100/4742 (86.5%)
  Progress: 4200/4742 (88.6%)
  Progress: 4300/4742 (90.7%)
  Progress: 4400/4742 (92.8%)
  Progress: 4500/4742 (94.9%)
  Progress: 4600/4742 (97.0%)
  [WARN] row 4645 failed after 3 attempts: Invalid \escape: line 1 column 13 (char 12)
  Progress: 4700/4742 (99.1%)

Done.  Written: 4727  |  Failed: 15  |  Skipped (checkpoint): 0

'''