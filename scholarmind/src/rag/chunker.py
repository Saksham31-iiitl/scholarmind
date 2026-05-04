"""Chunk papers into overlapping windows for retrieval."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    text: str
    position: int
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_text(text: str, window: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if len(words) <= window:
        return [text]
    chunks = []
    step = window - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + window])
        chunks.append(chunk)
        if i + window >= len(words):
            break
    return chunks


def chunk_papers(papers: Iterable[dict], window: int = 256, overlap: int = 32) -> list[Chunk]:
    """Convert papers into Chunk objects (uses title + abstract; PDF parsing can extend this)."""
    out: list[Chunk] = []
    for p in papers:
        full = f"{p['title']}. {p['abstract']}"
        for i, c in enumerate(chunk_text(full, window, overlap)):
            out.append(Chunk(
                chunk_id=f"{p['id']}::{i}",
                paper_id=p["id"],
                title=p["title"],
                text=c,
                position=i,
                metadata={
                    "authors": p.get("authors", []),
                    "published": p.get("published"),
                    "url": p.get("url"),
                    "categories": p.get("categories", []),
                },
            ))
    return out
