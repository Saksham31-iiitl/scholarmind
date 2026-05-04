"""
LangGraph orchestration of all agents.

Flow:
   START → planner → retrieve → synthesize → critic
                                    ↑           │
                                    └───revise──┤
                                                ▼
                                            verify → END
"""
from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from src.agents.state import AgentState
from src.agents.planner import planner_node
from src.agents.retriever_agent import RetrieverAgent
from src.agents.synthesizer import synthesizer_node
from src.agents.critic import critic_node, should_revise
from src.agents.verifier import verifier_node
from src.utils.logging import logger


def build_graph():
    """Return a compiled LangGraph app."""
    retriever = RetrieverAgent()

    g = StateGraph(AgentState)
    g.add_node("plan", planner_node)
    g.add_node("retrieve", retriever)
    g.add_node("synthesize", synthesizer_node)
    g.add_node("critic", critic_node)
    g.add_node("verify", verifier_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", "critic")
    g.add_conditional_edges("critic", should_revise, {
        "synthesize": "synthesize",
        "verify": "verify",
    })
    g.add_edge("verify", END)

    app = g.compile()
    logger.info("LangGraph compiled")
    return app
