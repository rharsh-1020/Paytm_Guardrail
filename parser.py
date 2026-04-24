import hashlib
from langchain_core.documents import Document

BLOCK_PHRASE = "Please enable JavaScript to view the page content."


def extract_text_from_notifications(records):
    documents = []
    seen_hashes = set()

    for record in records:
        text = (record.get("text") or "").strip()
        if not text:
            continue
        if BLOCK_PHRASE.lower() in text.lower():
            continue

        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if doc_hash in seen_hashes:
            continue
        seen_hashes.add(doc_hash)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": record.get("url", ""),
                    "title": record.get("title", ""),
                    "date": record.get("date", ""),
                    "doc_id": record.get("doc_id", ""),
                    "pdf_url": record.get("pdf_url", ""),
                    "hash": doc_hash,
                },
            )
        )

    return documents


# Backward-compatible alias

def extract_text_from_pdfs(records):
    return extract_text_from_notifications(records)
