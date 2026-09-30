"""Transparent Top-K cosine-similarity retrieval."""

from dataclasses import dataclass

import numpy as np

from core.chunker import DocumentChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


def retrieve(
    query_vector: np.ndarray,
    chunk_vectors: np.ndarray,
    chunks: list[DocumentChunk],
    top_k: int = 4,
    threshold: float = 0.0,
    redundancy_threshold: float = 0.9,
) -> list[RetrievedChunk]:
    if top_k < 1:
        raise ValueError("Top-K must be at least 1.")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Similarity threshold must be between -1 and 1.")
    vectors = np.asarray(chunk_vectors, dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Chunk vectors and chunks must have matching rows.")
    if vectors.shape[1] != query.size:
        raise ValueError("Query and chunk embeddings must have matching dimensions.")
    if not chunks:
        return []

    norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []
    scores = (vectors @ query) / np.maximum(norms * query_norm, 1e-12)
    ranked = np.argsort(scores)[::-1]

    results: list[RetrievedChunk] = []
    for index in ranked:
        score = float(scores[index])
        if score < threshold:
            break
        candidate = chunks[int(index)]
        tokens = set(candidate.text.casefold().split())
        redundant = False
        for existing in results:
            existing_tokens = set(existing.chunk.text.casefold().split())
            union = tokens | existing_tokens
            similarity = len(tokens & existing_tokens) / len(union) if union else 1.0
            if similarity >= redundancy_threshold:
                redundant = True
                break
        if not redundant:
            results.append(RetrievedChunk(candidate, score))
        if len(results) == top_k:
            break
    return results