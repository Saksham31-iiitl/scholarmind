"""LLM-powered concept & claim extraction (for KG ingestion)."""
from __future__ import annotations
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils.llm_factory import get_llm
from src.utils.logging import logger

SYSTEM = """You are an information-extraction model for academic papers.
Given a paper title and abstract, return STRICT JSON with two keys:
  "concepts": list of 3-8 lower-case noun-phrase technical concepts (e.g. "graph neural networks").
  "claims":  list of 2-5 self-contained factual claims made by the paper.
Return ONLY the JSON object, no prose."""


def _safe_json(s: str) -> dict:
    """Extract JSON from a model response that may contain code fences."""
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in: {s[:200]}")
    return json.loads(m.group(0))


def extract(title: str, abstract: str) -> dict:
    """Return {'concepts':[...], 'claims':[...]} for a paper."""
    llm = get_llm(temperature=0.0, max_tokens=512)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"Title: {title}\n\nAbstract: {abstract}"),
    ]
    resp = llm.invoke(msgs).content
    try:
        return _safe_json(resp)
    except Exception as e:
        logger.warning(f"Extraction failed: {e}; returning empty.")
        return {"concepts": [], "claims": []}
