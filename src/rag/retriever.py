"""
Hybrid retriever: BM25 (sparse) + FAISS (dense) + cross-encoder reranking.

This is the core of the RAG pipeline. It implements:
  1. Sparse retrieval via BM25Okapi  (good for exact-term matches)
  2. Dense retrieval via Sentence-Transformers + FAISS IndexFlatIP (semantic)
  3. Reciprocal Rank Fusion (RRF) of the two ranked lists
  4. Cross-encoder reranking of top-N for final precision
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Sequence
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.rag.chunker import Chunk
from src.utils.config import settings
from src.utils.logging import logger


class HybridRetriever:
    """Hybrid sparse + dense retriever with cross-encoder reranking."""

    def __init__(
        self,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
    ) -> None:
        self._embedding_model_name = embedding_model or settings.embedding_model
        self._reranker_model_name = reranker_model or settings.reranker_model
        self._embedder: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.faiss_index: faiss.Index | None = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(self._reranker_model_name)
        return self._reranker

    # ------------------------- index build -------------------------
    def build(self, chunks: Sequence[Chunk]) -> None:
        """Build BM25 + FAISS indices over chunks."""
        if not chunks:
            raise ValueError("Cannot build retriever on empty chunk list")
        self.chunks = list(chunks)
        logger.info(f"Building BM25 over {len(chunks)} chunks…")
        tokenized = [c.text.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        logger.info("Encoding chunks for dense index…")
        embeddings = self.embedder.encode(
            [c.text for c in self.chunks],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        logger.info(f"FAISS index dim={dim} ntotal={self.faiss_index.ntotal}")

    # ------------------------- persistence -------------------------
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(path / "index.faiss"))
        with (path / "store.pkl").open("wb") as f:
            pickle.dump({"chunks": [c.to_dict() for c in self.chunks], "bm25": self.bm25}, f)
        logger.info(f"Saved retriever → {path}")

    def load(self, path: Path | str) -> "HybridRetriever":
        path = Path(path)
        self.faiss_index = faiss.read_index(str(path / "index.faiss"))
        with (path / "store.pkl").open("rb") as f:
            data = pickle.load(f)
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self.bm25 = data["bm25"]
        logger.info(f"Loaded retriever ← {path} ({len(self.chunks)} chunks)")
        return self

    # ------------------------- retrieval ---------------------------
    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(query.lower().split()))

    def _dense_scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, _ = self.faiss_index.search(q, len(self.chunks))
        # scores shape (1, N); reorder to align with chunk indices
        sims = np.zeros(len(self.chunks))
        _, idx = self.faiss_index.search(q, len(self.chunks))
        sims[idx[0]] = scores[0]
        return sims

    @staticmethod
    def _rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
        """Reciprocal Rank Fusion."""
        scores: dict[int, float] = {}
        for rl in rank_lists:
            for rank, idx in enumerate(rl):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        return scores

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        """Return reranked (chunk, score) pairs."""
        top_k_retrieve = settings.top_k_retrieve
        top_k_rerank = top_k or settings.top_k_rerank

        bm25 = self._bm25_scores(query)
        dense = self._dense_scores(query)
        bm25_top = np.argsort(-bm25)[:top_k_retrieve].tolist()
        dense_top = np.argsort(-dense)[:top_k_retrieve].tolist()

        fused = self._rrf([bm25_top, dense_top])
        candidate_ids = sorted(fused, key=fused.get, reverse=True)[:top_k_retrieve]

        if not settings.use_reranker:
            return [(self.chunks[i], fused[i]) for i in candidate_ids[:top_k_rerank]]

        pairs = [(query, self.chunks[i].text) for i in candidate_ids]
        ce_scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidate_ids, ce_scores), key=lambda x: -x[1])[:top_k_rerank]
        return [(self.chunks[i], float(s)) for i, s in ranked]
