"""CPU-first local sentence-transformer embeddings."""

from collections.abc import Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when the local embedding model is not prepared on this machine."""


class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, model=None):
        self.model_name = model_name
        self._model = model

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_name,
                    device="cpu",
                    local_files_only=True,
                )
            except Exception as exc:
                raise EmbeddingModelUnavailable(
                    "The local embedding model is not cached. Prepare it explicitly with "
                    "`hf download sentence-transformers/all-MiniLM-L6-v2`, then retry. "
                    "For offline workshops, run that command before class."
                ) from exc
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._load_model().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Embedding model returned an unexpected vector shape.")
        return vectors