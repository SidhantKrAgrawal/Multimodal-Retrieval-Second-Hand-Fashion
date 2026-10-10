import sys
import time
import random
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import undetected_chromedriver as uc
from bs4 import BeautifulSoup

html_file = "./hrefs/womens_trousers.html"
BASE_URL = "https://shpock.com"
MAX_LISTINGS = 450
THREADS = 2
CSV_FILE = "shpock_listings.csv"

csv_lock = threading.Lock()
driver_lock = threading.Lock()
counter_lock = threading.Lock()
scraped_count = 0

def initialize_csv():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "Description", "Image URL", "Style", "Size", "Brand", "Colour", "Condition", "URL"])

def write_to_csv(data):
    with csv_lock:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                data["title"],
                data["description"],
                data["image_url"],
                data["Style"],
                data["Size"],
                data["Brand"],
                data["Colour"],
                data["Condition"],
                data["url"]
            ])
            f.flush()  # Ensure the data is written immediately

def get_undetected_driver():
    # serialise instantiation to avoid concurrent chromedriver file conflicts
    with driver_lock:
        try:
            return uc.Chrome(headless=False)
        except FileExistsError as e:
            problematic_file = os.path.join(os.environ["APPDATA"], "undetected_chromedriver", "undetected_chromedriver.exe")
            if os.path.exists(problematic_file):
                print("Removing existing driver file to resolve FileExistsError")
                os.remove(problematic_file)
            return uc.Chrome(headless=False)

def extract_product_links():
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return [BASE_URL + a["href"] for a in soup.find_all("a", href=True)
            if "/en-gb/i/" in a["href"]]

def scrape_with_driver(driver, url):
    try:
        print(f"Processing: {url}")
        driver.get(url)
        time.sleep(random.uniform(1, 3))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        title_tag = soup.find("title")
        title = title_tag.text.split("|")[0].strip() if title_tag else "N/A"
        
        description_meta = soup.find("meta", {"name": "description"})
        description = description_meta["content"] if description_meta else "N/A"
        
        image_meta = soup.find("meta", {"property": "og:image"})
        image_url = image_meta["content"] if image_meta else "N/A"
        
        attributes = {}
        for div in soup.find_all("div", class_="Box-sc-1u5jpde-0 hEAnCG"):
            key_span = div.find("span", class_="Text-sc-12c3ify-0 iamBVO")
            value_span = div.find("span", class_="Text-sc-12c3ify-0 fZdSBc")
            if key_span and value_span:
                attributes[key_span.text.strip().replace(":", "")] = value_span.text.strip()

        print(f" {title}  | Brand: {attributes.get('Brand', 'N/A')}  | "
              f"Colour: {attributes.get('Colour', 'N/A')}  | "
              f"Condition: {attributes.get('Condition', 'N/A')}")
        
        return {
            "title": title,
            "description": description,
            "image_url": image_url,
            "Style": attributes.get("Style", "N/A"),
            "Size": attributes.get("Size", "N/A"),
            "Brand": attributes.get("Brand", "N/A"),
            "Colour": attributes.get("Colour", "N/A"),
            "Condition": attributes.get("Condition", "N/A"),
            "url": url
        }
    except Exception as e:
        print(f" Error fetching {url}: {e}")
        return None

def worker(urls):
    global scraped_count
    driver = get_undetected_driver()
    for url in urls:
        data = scrape_with_driver(driver, url)
        if data:
            write_to_csv(data)
            with counter_lock:
                scraped_count += 1
                print(f"Scraped {scraped_count} out of {MAX_LISTINGS}")
    try:
        driver.quit()
    except Exception as e:
        print("Error quitting driver:", e)

def partition(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def main():
    initialize_csv()
    product_links = extract_product_links()[:MAX_LISTINGS]
    print(f" Scraping {len(product_links)} listings with {THREADS} persistent browsers.")
    chunks = partition(product_links, THREADS)
    
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(worker, chunk) for chunk in chunks]
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
