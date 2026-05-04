"""Planner agent: decomposes a research question into focused sub-questions."""
from __future__ import annotations
import json, re
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.state import AgentState
from src.utils.llm_factory import get_llm
from src.utils.logging import logger

SYSTEM = """You are a research planner. Decompose the user's research question into
3-5 focused sub-questions that, taken together, would answer it comprehensively.
Return STRICT JSON: {"sub_questions": ["...", "..."]}"""


def planner_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.1, max_tokens=512)
    resp = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=state["query"]),
    ]).content
    m = re.search(r"\{.*\}", resp, re.DOTALL)
    sub_qs = json.loads(m.group(0))["sub_questions"] if m else [state["query"]]
    logger.info(f"Planner produced {len(sub_qs)} sub-questions")
    return {"sub_questions": sub_qs, "iteration": 0}
