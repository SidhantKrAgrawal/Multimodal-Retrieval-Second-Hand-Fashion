import os
import json

labels_path = "../dataset/labels"
output_file = "captions.txt"

json_files = [f for f in os.listdir(labels_path) if f.endswith(".json")]

def colors_to_sentence(colors):
    if not colors:
        return ""
    # "Multicolor" dominates a 2-color list but is dropped when more specific colors exist
    if "Multicolor" in colors:
        if len(colors) == 2:
            return "Multicolor"
        elif len(colors) > 2:
            colors.remove("Multicolor")
    if len(colors) == 1:
        return colors[0]
    elif len(colors) == 2:
        return f"{colors[0]} and {colors[1]}"
    else:
        return f"{', '.join(colors[:-1])} and {colors[-1]}"

def remove_duplicates(sentence):
    words = sentence.split()
    seen = set()
    unique_words = [word for word in words if not (word.lower() in seen or seen.add(word.lower()))]
    return " ".join(unique_words)

with open(output_file, "w") as f_out:
    if not json_files:
        print("No JSON files found in the directory.")
    else:
        for json_file in json_files:
            file_path = os.path.join(labels_path, json_file)
            with open(file_path, "r") as f:
                data = json.load(f)

            category = data.get("category", "").strip()
            trend = data.get("trend", "").strip()
            pattern = data.get("pattern", "").strip()
            cut = data.get("cut", [])
            type_ = data.get("type", "").strip()
            colors = colors_to_sentence(data.get("colors", []))
            season = data.get("season", "").strip()

            if category.lower() in ["men", "children"]:
                category = f"{category}'s"
            elif category.lower() == "unisex":
                category = ""

            if isinstance(cut, list):
                cut = [c for c in cut if c.lower() not in ["none", "regular"]]
                cut = cut[0] if cut else ""

            if pattern.lower() == "other":
                pattern = ""

            sentence = f"{category} {trend} {pattern} {cut} {type_} in {colors}"
            if season.lower() in ["winter", "summer"]:
                sentence += f", ideal for {season}"

            sentence = remove_duplicates(sentence.replace("None", "").strip())
            sentence = " ".join(sentence.split()).strip(", ")

            formatted_filename = json_file.replace("labels_", "").replace(".json", "")
            f_out.write(f"{formatted_filename}: {sentence}\n")
            print(f"{formatted_filename}: {sentence}")

print(f"\nCaptions saved to {output_file}")
