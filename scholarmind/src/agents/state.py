"""Shared TypedDict state passed between LangGraph nodes."""
from __future__ import annotations
from typing import Annotated, TypedDict
from operator import add


class AgentState(TypedDict, total=False):
    query: str                         # user research question
    sub_questions: list[str]           # decomposed by Planner
    retrieved: list[dict]              # [{chunk_id, text, score, paper_id, title, url}]
    draft: str                         # synthesizer output
    critique: str                      # critic feedback
    iteration: int                     # critic loop counter
    verified_citations: list[dict]     # citation grounding output
    final_answer: str                  # final survey
    messages: Annotated[list, add]     # accumulating message log
