"""Smoke tests; CI-friendly (no LLM/network calls)."""
from src.rag.chunker import chunk_text, chunk_papers


def test_chunk_text_short():
    out = chunk_text("hello world", window=512)
    assert out == ["hello world"]


def test_chunk_text_overlap():
    text = " ".join([f"w{i}" for i in range(300)])
    out = chunk_text(text, window=100, overlap=20)
    assert len(out) > 1
    # overlap check
    a = out[0].split()[-20:]
    b = out[1].split()[:20]
    assert a == b


def test_chunk_papers():
    papers = [{
        "id": "1234.5678", "title": "T", "abstract": "A " * 600,
        "authors": ["Doe, J."], "url": "u", "categories": ["cs.LG"],
    }]
    chunks = chunk_papers(papers, window=100, overlap=10)
    assert all(c.paper_id == "1234.5678" for c in chunks)
    assert len(chunks) > 1
