"""Critic agent: scores draft and emits actionable feedback (Reflexion pattern)."""
from __future__ import annotations
import json, re
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.state import AgentState
from src.utils.llm_factory import get_llm
from src.utils.config import settings
from src.utils.logging import logger

SYSTEM = """You are a strict peer-reviewer. Evaluate the draft on:
  - Faithfulness: every claim grounded in sources?
  - Coverage: are all sub-questions addressed?
  - Structure: clear sections + citations?
  - Conciseness: any padding / repetition?

Return STRICT JSON:
{
  "score": 0-10,
  "issues": ["..."],
  "needs_revision": true|false
}"""


def critic_node(state: AgentState) -> AgentState:
    iteration = state.get("iteration", 0) + 1
    user = (
        f"Sub-questions: {state.get('sub_questions')}\n\n"
        f"Draft:\n{state['draft']}\n\n"
        f"Sources: {len(state['retrieved'])} chunks supplied."
    )
    llm = get_llm(temperature=0.0, max_tokens=512)
    resp = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)]).content
    m = re.search(r"\{.*\}", resp, re.DOTALL)
    parsed = json.loads(m.group(0)) if m else {"score": 7, "issues": [], "needs_revision": False}
    logger.info(f"Critic iter={iteration} score={parsed.get('score')} revise={parsed.get('needs_revision')}")
    if iteration >= settings.max_critic_iterations:
        parsed["needs_revision"] = False
    return {
        "critique": json.dumps(parsed),
        "iteration": iteration,
    }


def should_revise(state: AgentState) -> str:
    """LangGraph conditional edge function."""
    try:
        c = json.loads(state["critique"])
        return "synthesize" if c.get("needs_revision") else "verify"
    except Exception:
        return "verify"
