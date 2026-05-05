"""
Citation Verifier: parses [P1], [P2], ... tokens from the draft and verifies that
each cited claim is supported by the retrieved chunk via embedding similarity.

Removes or flags hallucinated citations.
"""
from __future__ import annotations
import re
import numpy as np
from src.agents.state import AgentState
from src.utils.config import settings
from src.utils.logging import logger

CITATION_RE = re.compile(r"\[P(\d+)\]")
SIM_THRESHOLD = 0.45


def _split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def verifier_node(state: AgentState) -> AgentState:
    if settings.use_dense:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(settings.embedding_model)
    else:
        encoder = None
    sources = state["retrieved"]
    draft = state["draft"]

    sentences = _split_into_sentences(draft)
    verified: list[dict] = []
    cleaned_lines: list[str] = []

    for sent in sentences:
        cite_ids = CITATION_RE.findall(sent)
        if not cite_ids:
            cleaned_lines.append(sent)
            continue
        ok_ids: list[str] = []
        for cid in cite_ids:
            idx = int(cid) - 1
            if not (0 <= idx < len(sources)):
                continue
            if encoder is not None:
                sent_vec = encoder.encode([sent], normalize_embeddings=True)[0]
                src_vec = encoder.encode([sources[idx]["text"]], normalize_embeddings=True)[0]
                sim = float(np.dot(sent_vec, src_vec))
                supported = sim >= SIM_THRESHOLD
            else:
                sim = 1.0
                supported = True
            verified.append({
                "sentence": sent,
                "citation": f"P{cid}",
                "paper_id": sources[idx]["paper_id"],
                "similarity": sim,
                "supported": supported,
            })
            if supported:
                ok_ids.append(cid)

        if ok_ids:
            kept = re.sub(CITATION_RE, lambda m: f"[P{m.group(1)}]" if m.group(1) in ok_ids else "", sent)
            cleaned_lines.append(kept)
        else:
            cleaned_lines.append(sent + " *[citation unverified]*")

    final = " ".join(cleaned_lines)

    # Append sources section
    final += "\n\n## References\n"
    for i, s in enumerate(sources, 1):
        authors = ", ".join(s.get("authors", [])[:3])
        final += f"[P{i}] {authors}. *{s['title']}*. {s.get('url', '')}\n"

    n_total = len(verified)
    n_supp = sum(1 for v in verified if v["supported"])
    logger.info(f"Verifier: {n_supp}/{n_total} citations supported")
    return {"verified_citations": verified, "final_answer": final}
