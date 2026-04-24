from scrapy.crawler import CrawlerProcess
from scrapy import signals
from scraper import RBISpider
from parser import extract_text_from_notifications
from embeddings import chunk_documents, get_embeddings
from vectorstore import store_in_chroma
import os


def disable_broken_local_proxy():
    """
    Some environments set proxy vars to 127.0.0.1:9 (discard port),
    which causes outbound HTTPS requests to fail with WinError 10061.
    Only strip this known-bad proxy pattern.
    """
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ]

    for key in proxy_keys:
        value = os.environ.get(key)
        if value and ("127.0.0.1:9" in value or "localhost:9" in value):
            os.environ.pop(key, None)


def run_pipeline():
    disable_broken_local_proxy()
    print("[START] Starting RBI ingestion pipeline")

    process = CrawlerProcess()
    items = []

    def collect_item(item):
        items.append(item)

    process.crawl(RBISpider)
    for crawler in process.crawlers:
        crawler.signals.connect(collect_item, signal=signals.item_scraped)

    process.start()

    if not items:
        print("No documents scraped.")
        return

    docs = extract_text_from_notifications(items)

    if not docs:
        print("No valid documents.")
        return

    chunks = chunk_documents(docs)
    embeddings = get_embeddings()

    db = store_in_chroma(chunks, embeddings)

    results = db.similarity_search("wallet limit", k=2)

    print("\n[SEARCH] TEST QUERY RESULTS:")
    for r in results:
        title = (r.metadata.get("title") or "").encode("ascii", errors="replace").decode("ascii")
        source = (r.metadata.get("source") or "").encode("ascii", errors="replace").decode("ascii")
        snippet = r.page_content[:200].encode("ascii", errors="replace").decode("ascii")
        print(f"Title: {title}")
        print(f"Source: {source}")
        print(snippet, "\n")


if __name__ == "__main__":
    run_pipeline()
