import csv
import os
import requests

def download_image(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded image to {output_path}")
    else:
        print(f"Failed to download image from {url}: HTTP {response.status_code}")

def main():
    csv_file = "../dataset2/labels.csv"
    images_dir = "../dataset2/images"

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    index = 0
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_url = row.get("Image URL", "").strip()
            output_file = os.path.join(images_dir, f"{index}.jpg")
            if image_url == "N/A" or image_url == "":
                print(f"Skipping index {index} (image URL is N/A)")
            else:
                download_image(image_url, output_file)
            index += 1

    print(f"\nProcessed {index} entries from the CSV.")

if __name__ == "__main__":
    main()
