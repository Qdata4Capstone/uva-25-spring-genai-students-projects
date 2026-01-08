import requests
from bs4 import BeautifulSoup
from newspaper import Article
from fake_useragent import UserAgent
import json
import time
import random
from urllib.parse import urljoin
from langdetect import detect
from tqdm import tqdm
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ua = UserAgent()

def extract_links(url, response_text, language):
    """
    根据页面内容和语言，提取页面中可能的链接。
    除了各网站已有的规则，还针对中文网站增加备用选择器，以覆盖更多可能的文章链接。
    """
    try:
        soup = BeautifulSoup(response_text, "lxml")
        links = []
        base_url = "/".join(url.split("/")[:3])
        
        # 针对指定网站，采用专门的 CSS 选择器：
        if "people.com.cn" in url:
            links = [a["href"] for a in soup.select("div.rm_relevant a[href*='/n1/202'], div.rm_ranking a[href*='/n1/202']") if a.has_attr("href") and a["href"].endswith(".html")]
        elif "xinhuanet.com" in url:
            links = [a["href"] for a in soup.select("a[href*='content_']") if a.has_attr("href")]
        elif "thepaper.cn" in url:
            links = [a["href"] for a in soup.select("a[href*='newsDetail_']") if a.has_attr("href")]
        elif "nytimes.com" in url:
            links = [a["href"] for a in soup.select("a[href*='/202']") if a.has_attr("href")]
        elif "cnn.com" in url:
            links = [a["href"] for a in soup.select("a.container__link[href*='/202']") if a.has_attr("href")]
        elif "asahi.com" in url:
            links = [a["href"] for a in soup.select("a[href*='articles/']") if a.has_attr("href")]
        elif "nhk.or.jp" in url:
            links = [a["href"] for a in soup.select("a[href*='/news/html']") if a.has_attr("href")]
        elif "lemonde.fr" in url or "lefigaro.fr" in url:
            links = [a["href"] for a in soup.select("a[href*='/202']") if a.has_attr("href")]
        elif "elpais.com" in url or "elmundo.es" in url:
            links = [a["href"] for a in soup.select("a[href*='/202']") if a.has_attr("href")]
        elif "ifeng.com" in url:
            links = [a["href"] for a in soup.select("a[href*='/c/']") if a.has_attr("href")]
        else:
            # 针对其他网站，使用默认规则：
            links = [a["href"] for a in soup.select("a[href*='article'], a[href*='2025']") if a.has_attr("href")]
        
        # 针对中文站点，如果未获取到足够链接，尝试备用选择器
        if language == "zh" and not links:
            additional_selectors = [
                "a[href*='/news/']", 
                "a[href*='/article/']",
                "a[href*='content']"
            ]
            for sel in additional_selectors:
                links += [a["href"] for a in soup.select(sel) if a.has_attr("href")]
        
        # 转换为绝对 URL，并过滤部分包含排除关键词的链接
        links = [urljoin(base_url, href) for href in links]
        exclude_keywords = [
            "video", "interactive", "index.", "login", "signup", "jpg", "png",
            "twitter.com/intent", "facebook.com/sharer", "share", "comment", "subscribe",
            "home", "homepage", "category", "tag", "list", "archive"
        ]
        links = [href for href in links if not any(kw in href.lower() for kw in exclude_keywords)]
        
        # 尝试提取翻页链接
        next_page = soup.select_one("a.next, a[rel='next'], a[href*='page=']")
        if next_page and next_page.get("href"):
            next_url = urljoin(base_url, next_page["href"])
            links.append(next_url)
        
        logging.info(f"从 {url} 提取到 {len(links)} 个链接")
        return list(set(links))
    except Exception as e:
        logging.error(f"提取链接出错 {url}: {e}")
        return []

def crawl_article(url, language, output_file, pbar):
    """
    使用 requests 与 newspaper3k 抓取文章内容；若解析失败则尝试使用 Selenium 模拟浏览器加载。
    """
    if url.startswith("http://"):
        url = url.replace("http://", "https://")
    
    headers = {
        "User-Agent": ua.random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://www.google.com/"
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            response.raise_for_status()
            
            article = Article(url, memoize_articles=False)
            article.set_html(response.text)
            article.parse()
            
            if not article.title or len(article.title.strip()) < 3:
                logging.warning(f"标题为空或过短: {url}")
                try:
                    options = Options()
                    options.add_argument("--headless")
                    options.add_argument(f"user-agent={ua.random}")
                    options.add_argument("--no-sandbox")
                    options.add_argument("--disable-dev-shm-usage")
                    driver = webdriver.Chrome(options=options)
                    driver.get(url)
                    # 等待页面中 article 标签加载完成（可根据页面特点调整）
                    WebDriverWait(driver, 15).until(
                        lambda d: d.find_element(By.TAG_NAME, "article")
                    )
                    article.set_html(driver.page_source)
                    article.parse()
                    logging.info(f"Selenium 解析内容长度: {len(article.text)} - {url}")
                    driver.quit()
                except Exception as se:
                    logging.error(f"Selenium 失败 {url}: {se}")
                    return False
            
            if not article.title or len(article.title.strip()) < 3:
                logging.warning(f"Selenium 后标题仍为空或过短: {url}")
                return False
            
            if not article.text or len(article.text) < 50:
                logging.warning(f"文章内容太短或为空: {url}")
                return False
            
            data = {
                "title": article.title,
                "content": article.text,
                "date": str(article.publish_date) if article.publish_date else "unknown",
                "source": url.split("/")[2],
                "url": url,
                "language": language
            }
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            pbar.update(1)
            if pbar.n % 50 == 0:
                logging.info(f"[{language}] 已爬取 {pbar.n} 条新闻。最新一条: {data['title']} - {data['url']}")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"尝试 {attempt + 1}/{max_retries} 爬取文章出错 {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(2, 5))
            continue
    return False
