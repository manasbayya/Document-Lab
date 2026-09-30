"""Connect query embedding, cosine retrieval, context construction, and Ollama."""

from dataclasses import dataclass

import numpy as np

from core.chunker import DocumentChunk
from core.embeddings import EmbeddingService
from core.ollama_client import generate_answer
from core.retriever import RetrievedChunk, retrieve
from core.text_cleaner import clean_text

NOT_FOUND = "I couldn't find relevant information in the uploaded documents. Try a question related to their content."


@dataclass(frozen=True)
class RAGResult:
    answer: str
    sources: list[RetrievedChunk]


def build_context(sources: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[Source: {item.chunk.source}; page: {item.chunk.page_number or 'not available'}; "
        f"chunk: {item.chunk.chunk_id}]\n{item.chunk.text}"
        for item in sources
    )


def answer_question(
    question: str,
    chunks: list[DocumentChunk],
    chunk_vectors: np.ndarray,
    embedding_service: EmbeddingService,
    model: str,
    top_k: int = 4,
    threshold: float = 0.2,
    temperature: float = 0.2,
    ollama_url: str | None = None,
) -> RAGResult:
    cleaned_question = clean_text(question)
    if not cleaned_question:
        raise ValueError("Enter a question before sending it.")
    if not chunks:
        raise ValueError("Upload and index a document before asking a question.")
    query_vector = embedding_service.encode([cleaned_question])[0]
    sources = retrieve(query_vector, chunk_vectors, chunks, top_k, threshold)
    if not sources:
        return RAGResult(NOT_FOUND, [])
    kwargs = {"temperature": temperature}
    if ollama_url:
        kwargs["base_url"] = ollama_url
    answer = generate_answer(model, cleaned_question, build_context(sources), **kwargs)
    return RAGResult(answer, sources)