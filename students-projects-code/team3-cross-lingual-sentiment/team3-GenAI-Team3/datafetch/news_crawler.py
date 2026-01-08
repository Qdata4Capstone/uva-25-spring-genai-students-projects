from utils import extract_links, crawl_article, ua
import requests
from tqdm import tqdm
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_likely_article_url(url):
    """
    判断 URL 是否可能为文章页：
    - 排除首页、栏目页、列表页等常见非文章页面。
    - 对于中文网站，可放宽一些规则（首页和列表页会在 extract_links 过程中提取出真实文章链接）
    """
    exclude_patterns = [
        "/index.", "/home", "/category", "/tag", "/list", "/archive",
        "/page=", "/search", "/about", "/contact"
    ]
    # 对于部分中文网站，如果 URL 仅为域名或根路径，则认为是列表页，不直接作为文章页解析
    parsed = requests.utils.urlparse(url)
    if parsed.path in ["", "/"]:
        return False

    include_patterns = [
        "/c/", "/article/", "/content_", "/newsDetail_",  # 常见文章页模式
        r"\d{8}"  # 包含日期格式，如 20250412
    ]
    url_lower = url.lower()
    if any(p in url_lower for p in exclude_patterns):
        return False
    if any(p in url_lower for p in include_patterns) or url.endswith(".html"):
        return True
    return False

def crawl_language(language, site_urls, output_file, max_articles=10000):
    pbar = tqdm(total=max_articles, desc=f"爬取 {language} 新闻")
    visited_urls = set()
    link_queue = []
    
    # 初始化队列：从配置中提取该语言下所有的起始链接
    for site_url in site_urls[language]["urls"]:
        if site_url not in visited_urls:
            link_queue.append(site_url)
            visited_urls.add(site_url)

    # 定义一些基础请求头，部分网站可能需要较完整的请求头才能获得正常响应
    base_headers = {
        "User-Agent": ua.random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://www.google.com/"
    }
    
    while link_queue and pbar.n < max_articles:
        url = link_queue.pop(0)
        try:
            if url.startswith("http://"):
                url = url.replace("http://", "https://")
            response = requests.get(url, headers=base_headers, timeout=15)
            response.raise_for_status()
            
            # 尝试提取页面中的链接
            links = extract_links(url, response.text, language)
            for link in links:
                if link not in visited_urls and link not in link_queue:
                    link_queue.append(link)
                    visited_urls.add(link)
            
            # 如果当前 URL 本身符合文章页规则，则抓取文章内容
            if is_likely_article_url(url):
                crawl_article(url, language, output_file, pbar)
            else:
                logging.info(f"跳过非文章页: {url}")
        except Exception as e:
            logging.error(f"爬取站点出错 {url}: {e}")
    
    pbar.close()

if __name__ == "__main__":
    from config import site_urls
    if len(sys.argv) < 2:
        print("Usage: python main.py <language>")
        sys.exit(1)
    language = sys.argv[1]
    output_file = f"news_data_{language}.jsonl"
    crawl_language(language, site_urls, output_file)
