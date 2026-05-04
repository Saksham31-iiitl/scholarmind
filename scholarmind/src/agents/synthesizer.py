"""Synthesizer agent: writes a draft literature review with inline [Pn] citations."""
from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.state import AgentState
from src.utils.llm_factory import get_llm
from src.utils.logging import logger

SYSTEM = """You are a senior researcher writing a literature review section.

RULES:
1. Use ONLY the provided sources. Do NOT invent facts.
2. Cite every claim inline using [P1], [P2], … keyed to the source list.
3. Structure: Introduction → Themes (2-4) → Open Problems → Conclusion.
4. Be concise, technical, and use Markdown.
5. If a sub-question lacks evidence, say so explicitly."""


def synthesizer_node(state: AgentState) -> AgentState:
    sources_md = "\n\n".join(
        f"[P{i+1}] (paper_id={h['paper_id']}) {h['title']}\n{h['text']}"
        for i, h in enumerate(state["retrieved"])
    )
    user = (
        f"Research question: {state['query']}\n\n"
        f"Sub-questions: {state.get('sub_questions', [])}\n\n"
        f"Critique to address (if any): {state.get('critique', 'None')}\n\n"
        f"Sources:\n{sources_md}"
    )
    llm = get_llm(temperature=0.3, max_tokens=3000)
    draft = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)]).content
    logger.info(f"Synthesizer produced {len(draft.split())}-word draft")
    return {"draft": draft}
