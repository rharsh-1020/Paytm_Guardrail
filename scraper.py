import re
import scrapy
from urllib.parse import parse_qs, urlparse

BLOCK_PHRASE = "Please enable JavaScript to view the page content."


class RBISpider(scrapy.Spider):
    name = "rbi_spider"
    start_urls = ["https://www.rbi.org.in/Scripts/NotificationUser.aspx"]

    custom_settings = {
        "ROBOTSTXT_OBEY": False,
        "DOWNLOAD_DELAY": 1,
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seen_links = set()

    def parse(self, response):
        self.logger.info("Parsing RBI notification index page...")

        if BLOCK_PHRASE.lower() in response.text.lower():
            self.logger.warning("Blocked index page response; skipping crawl.")
            return

        links = response.css('a[href*="NotificationUser.aspx?Id="]::attr(href)').getall()
        for link in links:
            full_url = response.urljoin(link)
            if full_url in self._seen_links:
                continue
            self._seen_links.add(full_url)
            yield response.follow(full_url, callback=self.parse_detail)

    def parse_detail(self, response):
        if BLOCK_PHRASE.lower() in response.text.lower():
            self.logger.warning("Blocked detail page: %s", response.url)
            return

        row = response.css("tr.tablecontent2")
        if not row:
            self.logger.warning("Notification body container not found: %s", response.url)
            return

        paragraphs = row.css("p")
        lines = []
        for p in paragraphs:
            line = self._normalize_ws(" ".join(p.css("::text").getall()))
            if line:
                lines.append(line)

        body_text = "\n".join(lines).strip()
        if not body_text:
            self.logger.warning("No usable text extracted from: %s", response.url)
            return

        title = self._normalize_ws(row.css("p.head::text").get(default=""))
        if not title:
            title = self._normalize_ws(" ".join(response.css("title::text").getall()))

        date = self._extract_date(body_text)
        doc_id = self._extract_doc_id(response.url)
        raw_pdf_url = row.css('a[href*=".pdf"]::attr(href)').get(default="")
        pdf_url = response.urljoin(raw_pdf_url) if raw_pdf_url else ""

        yield {
            "doc_id": doc_id,
            "url": response.url,
            "title": title,
            "date": date,
            "text": body_text,
            "pdf_url": pdf_url,
        }

    @staticmethod
    def _normalize_ws(value):
        return " ".join(value.split()).strip()

    @staticmethod
    def _extract_doc_id(url):
        query = parse_qs(urlparse(url).query)
        raw_id = (query.get("Id") or [""])[0]
        return f"RBI_{raw_id}" if raw_id else f"RBI_{abs(hash(url))}"

    @staticmethod
    def _extract_date(text):
        match = re.search(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            text,
            flags=re.IGNORECASE,
        )
        return match.group(0) if match else ""
