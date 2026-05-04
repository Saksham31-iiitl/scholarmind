# ScholarMind — Architecture

## End-to-End Flow

```
USER QUERY
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                     │
│                                                                │
│   ┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌──────┐ │
│   │ Planner │ →  │Retriever │ →  │ Synthesizer │ →  │Critic│ │
│   └─────────┘    └────┬─────┘    └──────┬──────┘    └───┬──┘ │
│                       │                  ▲               │    │
│                       │                  └── reflexion ──┘    │
│                       ▼                                        │
│                 ┌──────────┐                                   │
│                 │ Verifier │                                   │
│                 └────┬─────┘                                   │
│                      ▼                                         │
│                  FINAL ANSWER                                  │
└──────────────────────────────────────────────────────────────┘
        ▲                        ▲
        │                        │
   ┌────┴─────┐            ┌─────┴─────┐
   │ Hybrid   │            │  Neo4j    │
   │  RAG     │            │   KG      │
   │(BM25+    │            └───────────┘
   │ FAISS+   │
   │ CrossEnc)│
   └──────────┘
```

## Agent Roles

| Agent | Responsibility | Output |
|---|---|---|
| Planner | Decompose query | sub_questions[] |
| Retriever | Hybrid RAG per sub-question | retrieved_chunks[] |
| Synthesizer | Generate draft with [Pn] citations | draft (markdown) |
| Critic | Score & critique (Reflexion) | feedback, needs_revision |
| Verifier | Embedding-similarity citation grounding | verified_citations[] |

## Hybrid Retrieval Algorithm

1. BM25Okapi → top 20 sparse hits
2. all-mpnet-base-v2 + FAISS IndexFlatIP → top 20 dense hits
3. **Reciprocal Rank Fusion** (k=60) merges both lists
4. Cross-encoder ms-marco-MiniLM-L-6-v2 → final top-5

## Citation Grounding

Each sentence containing `[Pn]` is embedded and compared against the n-th source
chunk via cosine similarity. Threshold = 0.45. Below threshold → citation
stripped and sentence flagged `*[citation unverified]*`.

## Reflexion Loop

The Critic agent emits a JSON verdict; if `needs_revision=true` and
`iteration < MAX_CRITIC_ITERATIONS (=3)`, the graph routes back to the
Synthesizer with the critique injected. This implements the
**Reflexion** pattern (Shinn et al., 2023).
