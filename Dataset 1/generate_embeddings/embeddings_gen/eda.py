import os
import json
from collections import Counter

labels_path = "../dataset1/labels"

json_files = [f for f in os.listdir(labels_path) if f.endswith(".json")]
damage_counter = Counter()

for json_file in json_files:
    file_path = os.path.join(labels_path, json_file)
    with open(file_path, "r") as f:
        data = json.load(f)
    damage = data.get("damage", "")
    if damage:
        print(f"Label: {json_file}, Damage: {damage}")
        damage_counter[damage] += 1

sorted_damage = sorted(damage_counter.items(), key=lambda x: x[1], reverse=True)

print("\nUnique Damage Values Sorted by Frequency (Most to Least):")
for dmg, count in sorted_damage:
    print(f"{dmg}: {count}")
