"""Simple FAISS-backed vector store powered by the local BGE model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

try:  # pragma: no cover
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None
import numpy as np
try:  # pragma: no cover
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

from .config import settings

LOGGER = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    text: str
    metadata: dict


@dataclass
class BGEVectorStore:
    """Manages local BGE embeddings and FAISS search for semantic recall."""

    documents: List[VectorDocument] = field(default_factory=list)
    min_similarity: float = 0.25

    def __post_init__(self) -> None:
        try:
            LOGGER.info("Loading BGE model from %s", settings.bge.model_path)
            if SentenceTransformer is None or faiss is None:
                raise RuntimeError("SentenceTransformer/faiss unavailable")
            self._model = SentenceTransformer(settings.bge.model_path)
            dim = self._model.get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatIP(dim)
            self._refresh_index()
        except Exception as exc:  # pragma: no cover - fallback for missing model
            LOGGER.warning("BGE model unavailable, falling back to keyword recall: %s", exc)
            self._model = None
            self._index = None

    def _refresh_index(self) -> None:
        if not self.documents or self._model is None:
            return
        embeddings = self._embed([doc.text for doc in self.documents])
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("BGE model not loaded")
        vectors = self._model.encode(
            list(texts),
            batch_size=settings.bge.batch_size,
            normalize_embeddings=settings.bge.normalize_embeddings,
            convert_to_numpy=True,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors.astype("float32")

    def add(self, text: str, metadata: dict) -> None:
        self.documents.append(VectorDocument(text=text, metadata=metadata))
        if self._model is not None and self._index is not None:
            embedding = self._embed([text])
            self._index.add(embedding)

    def search(self, query: str, top_k: int = 5) -> List[VectorDocument]:
        if not self.documents:
            return []
        hits: List[Tuple[int, float]] = []
        if self._model is not None and self._index is not None:
            query_vec = self._embed([query])
            search_k = min(top_k, len(self.documents))
            scores, indices = self._index.search(query_vec, search_k)
            hits = [
                (idx, score)
                for idx, score in zip(indices[0], scores[0])
                if idx != -1 and score >= self.min_similarity
            ]
        if hits:
            ordered = sorted(hits, key=lambda item: item[1], reverse=True)[:top_k]
            return [self.documents[idx] for idx, _ in ordered]
        return self._keyword_fallback(query, top_k)

    @staticmethod
    def _keyword_match(query: str, text: str) -> int:
        overlap = set(query.split()) & set(text.split())
        return len(overlap)

    def _keyword_fallback(self, query: str, top_k: int) -> List[VectorDocument]:
        scored = []
        for doc in self.documents:
            score = self._keyword_match(query, doc.text)
            scored.append((doc, score))
        scored.sort(key=lambda item: (item[1], item[0].metadata.get("timestamp", 0.0)), reverse=True)
        hits = [doc for doc, score in scored if score > 0][:top_k]
        if hits:
            return hits
        # Final fallback: most recent documents
        return list(reversed(self.documents[-top_k:]))
