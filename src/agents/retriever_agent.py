"""Retriever agent: runs hybrid RAG against each sub-question."""
from __future__ import annotations
from src.agents.state import AgentState
from src.rag.retriever import HybridRetriever
from src.utils.config import settings
from src.utils.logging import logger


class RetrieverAgent:
    """Stateful wrapper so the index isn't reloaded each turn."""

    def __init__(self) -> None:
        self.retriever = HybridRetriever().load(settings.vector_store_path)

    def __call__(self, state: AgentState) -> AgentState:
        all_hits: list[dict] = []
        seen: set[str] = set()
        for q in state.get("sub_questions", [state["query"]]):
            for chunk, score in self.retriever.retrieve(q):
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                all_hits.append({
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "score": score,
                    "url": chunk.metadata.get("url"),
                    "authors": chunk.metadata.get("authors", []),
                })
        logger.info(f"Retriever fused {len(all_hits)} unique chunks")
        return {"retrieved": all_hits}
