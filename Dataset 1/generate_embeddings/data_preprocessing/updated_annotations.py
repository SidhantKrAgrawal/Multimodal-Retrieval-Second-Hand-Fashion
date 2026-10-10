import os
import json
import random

def get_condition_caption(condition, damage):
    if condition == 5:
        if random.random() < 0.5:
            options = [
                "Only worn once",
                "Only worn a few times",
                "Brand new",
                "excellent condition",
                "barely worn"
            ]
            return random.choice(options)
        else:
            return ""
    else:
        if damage:
            damage = damage.lower()
            if damage == "hole":
                options = [
                    "minor hole but apart from that item is in good shape",
                    "theres a small hole but its not too visible",
                    "has a small hole but otherwise good condition",
                    "has small hole"
                ]
                return random.choice(options)
            elif damage == "broken zipper":
                options = [
                    "zipper is broke",
                    "broken zipper but otherwise good condition",
                    "broken zipper but still wearable",
                    "zipper is broke however"
                ]
                return random.choice(options)
            elif damage == "missing buttons":
                options = [
                    "some buttons missing",
                    "missing buttons",
                    "has a few buttons missing"
                ]
                return random.choice(options)
            elif damage == "stains":
                options = [
                    "there are a few stains but will come out in the wash",
                    "a few minor stains but still good condition",
                    "there is some minor stainage but otherwise great condition",
                    "few stains"
                ]
                return random.choice(options)
        return ""


labels_path = "../dataset1/labels"
captions_file = "captions.txt"
output_file = "captions_updated.txt"

with open(captions_file, "r") as f:
    lines = f.readlines()

updated_lines = []
for line in lines:
    if ":" not in line:
        continue
    lhs, rhs = line.strip().split(":", 1)
    rhs = rhs.strip()

    json_filename = f"labels_{lhs}.json"
    json_path = os.path.join(labels_path, json_filename)

    if not os.path.exists(json_path):
        print(f"Warning: {json_filename} not found. Skipping...")
        updated_lines.append(line.strip())
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    size = data.get("size")
    if size and size.lower() != "none":
        # insert size after gendered prefix (e.g. "Men's") if present
        parts = rhs.split()
        if parts and "'" in parts[0]:
            parts.insert(1, f"size {size}")
        else:
            parts.insert(0, f"Size {size}")
        rhs = " ".join(parts)

    condition = data.get("condition")
    damage = data.get("damage", "")
    cond_desc = ""
    if condition is not None:
        cond_desc = get_condition_caption(condition, damage)
    if cond_desc:
        rhs = f"{rhs}. {cond_desc}"

    updated_lines.append(f"{lhs}: {rhs}")

with open(output_file, "w") as f:
    f.write("\n".join(updated_lines))

print(f"Updated captions saved to {output_file}")
