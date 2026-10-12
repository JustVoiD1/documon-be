
from typing import Any
from langchain_core.documents import Document

def sanitize_text(text: Any) -> str:
    """Removes NUL (0x00) characters which are illegal in PostgreSQL string/JSON fields."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return text.replace("\x00", "")


def sanitize_metadata(data: Any) -> Any:
    """Recursively removes NUL (0x00) characters from string values and dictionary keys."""
    if isinstance(data, str):
        return data.replace("\x00", "")
    elif isinstance(data, dict):
        return {
            (k.replace("\x00", "") if isinstance(k, str) else k): sanitize_metadata(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [sanitize_metadata(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(sanitize_metadata(item) for item in data)
    return data


def clean_document(doc: Document) -> Document:
    """Sanitizes page_content and metadata of a Document to prevent PostgreSQL 0x00 errors."""
    if hasattr(doc, "page_content") and isinstance(doc.page_content, str):
        doc.page_content = sanitize_text(doc.page_content)
    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
        doc.metadata = sanitize_metadata(doc.metadata)
    return doc
