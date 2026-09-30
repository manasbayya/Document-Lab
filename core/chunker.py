"""Character-window chunking with source metadata and configurable overlap."""

from dataclasses import dataclass
from hashlib import sha1

from core.document_loader import LoadedDocument


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    page_number: int | None


def create_chunks(
    document: LoadedDocument,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[DocumentChunk]:
    if chunk_size < 100:
        raise ValueError("Chunk size must be at least 100 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap
    for page in document.pages:
        text = page.text.strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                index = len(chunks) + 1
                digest = sha1(
                    f"{document.filename}:{page.page_number}:{index}:{piece}".encode()
                ).hexdigest()[:8]
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"chunk-{index:04d}-{digest}",
                        text=piece,
                        source=document.filename,
                        page_number=page.page_number,
                    )
                )
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks