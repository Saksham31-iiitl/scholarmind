"""Ingest papers from arXiv API and persist as JSONL."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import arxiv
from src.utils.config import settings
from src.utils.logging import logger


def fetch_arxiv(query: str, max_results: int = 50) -> list[dict]:
    """Fetch papers matching `query` from arXiv."""
    logger.info(f"Querying arXiv: '{query}' (max={max_results})")
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    # Use a slower page_size to avoid arXiv 429 rate-limiting
    client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=5)
    papers = []
    for result in client.results(search):
        papers.append({
            "id": result.entry_id.split("/")[-1],
            "title": result.title.strip(),
            "abstract": result.summary.strip().replace("\n", " "),
            "authors": [a.name for a in result.authors],
            "published": result.published.isoformat(),
            "categories": result.categories,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
        })
    logger.info(f"Fetched {len(papers)} papers")
    return papers


def save_jsonl(papers: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {path}")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--max", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else settings.raw_data_path / f"{args.query.replace(' ', '_')}.jsonl"
    papers = fetch_arxiv(args.query, args.max)
    save_jsonl(papers, out_path)
