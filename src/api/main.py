"""FastAPI service exposing the multi-agent research pipeline."""
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.pipelines.orchestrator import build_graph
from src.kg.graph import KnowledgeGraph
from src.utils.logging import logger


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=4, examples=["Survey GNNs for drug discovery"])


class QueryResponse(BaseModel):
    answer: str
    sub_questions: list[str]
    n_sources: int
    n_supported_citations: int
    n_total_citations: int


_graph = None  # populated on startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logger.info("Loading agents…")
    _graph = build_graph()
    yield
    logger.info("Shutdown.")


app = FastAPI(title="ScholarMind API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "graph_loaded": _graph is not None}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if _graph is None:
        raise HTTPException(503, "Graph not initialized")
    try:
        result = _graph.invoke({"query": req.question})
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(500, str(e))
    vc = result.get("verified_citations", [])
    return QueryResponse(
        answer=result["final_answer"],
        sub_questions=result.get("sub_questions", []),
        n_sources=len(result.get("retrieved", [])),
        n_supported_citations=sum(1 for v in vc if v["supported"]),
        n_total_citations=len(vc),
    )


@app.get("/kg/concept/{concept}")
def kg_lookup(concept: str, limit: int = 10):
    kg = KnowledgeGraph()
    try:
        return {"concept": concept, "papers": kg.find_related_papers(concept, limit)}
    finally:
        kg.close()


@app.get("/kg/summary")
def kg_summary():
    kg = KnowledgeGraph()
    try:
        return kg.graph_summary()
    finally:
        kg.close()
