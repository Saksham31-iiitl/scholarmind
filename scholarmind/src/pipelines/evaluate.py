"""RAGAS-based evaluation of the end-to-end pipeline."""
from __future__ import annotations
import yaml
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.pipelines.orchestrator import build_graph
from src.utils.logging import logger


def evaluate_dataset(yaml_path: str | Path) -> dict:
    """yaml file shape:
    questions:
      - q: "What are the main types of GNN aggregators?"
        ground_truth: "Mean, sum, max-pool, attention"
    """
    spec = yaml.safe_load(Path(yaml_path).read_text())
    app = build_graph()
    rows = []
    for item in spec["questions"]:
        result = app.invoke({"query": item["q"]})
        rows.append({
            "question": item["q"],
            "answer": result["final_answer"],
            "contexts": [c["text"] for c in result["retrieved"]],
            "ground_truth": item.get("ground_truth", ""),
        })
    ds = Dataset.from_list(rows)
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    logger.info(f"RAGAS scores: {scores}")
    return scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="configs/eval_set.yaml")
    args = parser.parse_args()
    evaluate_dataset(args.dataset)
