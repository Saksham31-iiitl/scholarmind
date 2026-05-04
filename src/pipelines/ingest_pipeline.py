"""End-to-end ingest pipeline: arXiv → chunk → index → KG."""
from __future__ import annotations
from pathlib import Path
from tqdm import tqdm

from src.rag.ingest import fetch_arxiv, save_jsonl, load_jsonl
from src.rag.chunker import chunk_papers
from src.rag.retriever import HybridRetriever
from src.kg.graph import KnowledgeGraph
from src.kg.extractor import extract
from src.utils.config import settings
from src.utils.logging import logger


def run(query: str, max_papers: int = 50, build_kg: bool = True) -> None:
    raw = settings.raw_data_path / f"{query.replace(' ', '_')}.jsonl"
    if not raw.exists():
        save_jsonl(fetch_arxiv(query, max_papers), raw)
    papers = load_jsonl(raw)

    # Build retriever
    chunks = chunk_papers(papers, window=256, overlap=32)
    retr = HybridRetriever()
    retr.build(chunks)
    retr.save(settings.vector_store_path)

    # Build KG
    if build_kg:
        try:
            kg = KnowledgeGraph()
            kg.init_schema()
            for p in tqdm(papers, desc="KG ingest"):
                ex = extract(p["title"], p["abstract"])
                kg.add_paper(p, ex.get("concepts", []), ex.get("claims", []))
            logger.info(f"KG summary: {kg.graph_summary()}")
            kg.close()
        except Exception as e:
            logger.warning(f"KG ingestion skipped: {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--max", type=int, default=50)
    p.add_argument("--no-kg", action="store_true")
    args = p.parse_args()
    run(args.query, args.max, build_kg=not args.no_kg)
