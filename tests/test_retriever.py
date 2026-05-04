"""Retriever build/retrieve smoke test (CPU-only, downloads small models on first run)."""
import pytest
from src.rag.chunker import Chunk
from src.rag.retriever import HybridRetriever


@pytest.fixture(scope="module")
def retriever():
    chunks = [
        Chunk("a::0", "a", "GNN survey",
              "Graph neural networks aggregate node features via message passing.",
              0, {}),
        Chunk("b::0", "b", "Vision transformers",
              "ViT applies attention over patches in images for classification.",
              0, {}),
        Chunk("c::0", "c", "Reinforcement learning",
              "PPO is a policy-gradient method for stable RL training.",
              0, {}),
    ]
    r = HybridRetriever()
    r.build(chunks)
    return r


def test_retrieves_relevant(retriever):
    hits = retriever.retrieve("graph neural networks")
    assert hits[0][0].paper_id == "a"


def test_retrieves_rl(retriever):
    hits = retriever.retrieve("policy gradient PPO")
    assert hits[0][0].paper_id == "c"
