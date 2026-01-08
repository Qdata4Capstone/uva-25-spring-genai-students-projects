import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import json
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from config import site_urls
import argparse  # Added for argument parsing

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}

def extract_article(url):
    try:
        response = requests.get(url, timeout=10, headers=HEADERS)
        if response.status_code != 200 or "text/html" not in response.headers.get("Content-Type", ""):
            return None

        soup = BeautifulSoup(response.content, "lxml")
        title = soup.title.text.strip() if soup.title else None

        paragraphs = soup.find_all("p")
        content = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        if not content or len(content) < 50:
            return None

        meta_time = soup.find("meta", {"property": "article:published_time"}) or \
                    soup.find("meta", {"name": "pubdate"}) or \
                    soup.find("meta", {"name": "date"})
        publish_time = meta_time["content"] if meta_time and meta_time.has_attr("content") else None

        return {
            "url": url,
            "title": title,
            "publish_time": publish_time,
            "content": content
        }
    except Exception:
        return None

def save_article(filepath, article, saved_urls):
    if article["url"] in saved_urls:
        return
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(article, ensure_ascii=False) + "\n")
    saved_urls.add(article["url"])

def producer(start_urls, link_queue, visited, lock, max_links):
    for start_url in start_urls:
        queue = [start_url]
        domain = start_url.split('/')[2]

        while queue and len(visited) < max_links:
            current_url = queue.pop(0)
            with lock:
                if current_url in visited or len(visited) >= max_links:
                    continue
                visited.add(current_url)

            try:
                response = requests.get(current_url, timeout=10, headers=HEADERS)
                if "text/html" not in response.headers.get("Content-Type", ""):
                    continue

                soup = BeautifulSoup(response.text, 'lxml')
                for a in soup.find_all("a", href=True):
                    href = urljoin(current_url, a['href'])
                    if href.startswith("http") and domain in href and "video" not in href:
                        with lock:
                            if href not in visited:
                                queue.append(href)
                                link_queue.put(href)
            except:
                continue

def consumer(link_queue, filepath, pbar, stop_event, saved_urls):
    while not stop_event.is_set():
        try:
            url = link_queue.get(timeout=3)
            article = extract_article(url)
            if article:
                save_article(filepath, article, saved_urls)
                pbar.update(1)
                if pbar.n >= pbar.total:
                    stop_event.set()
        except:
            continue

def process_language(lang, config):
    print(f"\n[+] Crawling {lang} - Target: {config['target_count']} articles")
    os.makedirs("output", exist_ok=True)
    filepath = f"output/{lang}_news.jsonl"

    link_queue = Queue()
    visited = set()
    saved_urls = set()
    lock = threading.Lock()
    stop_event = threading.Event()

    pbar = tqdm(total=config["target_count"], desc=f"{lang} downloaded", unit="article")

    producer_thread = threading.Thread(target=producer, args=(config["urls"], link_queue, visited, lock, config["target_count"]))
    producer_thread.start()

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(consumer, link_queue, filepath, pbar, stop_event, saved_urls) for _ in range(10)]

        producer_thread.join()
        stop_event.set()

    pbar.close()
    print(f"[✓] {lang} - Crawled {pbar.n} articles")

def main():
    parser = argparse.ArgumentParser(description="News crawler for multiple languages")
    parser.add_argument("--lang", choices=site_urls.keys(), help="Language to crawl (e.g., zh, en, ja, fr, es)")
    parser.add_argument("language", nargs="?", choices=site_urls.keys(), help="Language to crawl (positional, for SLURM compatibility)")
    args = parser.parse_args()

    # Use --lang if provided, otherwise fall back to positional language, or process all languages
    lang_to_crawl = args.lang or args.language
    if lang_to_crawl:
        if lang_to_crawl in site_urls:
            process_language(lang_to_crawl, site_urls[lang_to_crawl])
        else:
            print(f"Error: Language '{lang_to_crawl}' not found in config")
            exit(1)
    else:
        for lang, config in site_urls.items():
            process_language(lang, config)

if __name__ == "__main__":
    main()