"""Evaluation configuration — thresholds gate CI, models come from env."""

from pydantic import BaseModel


class EvalConfig(BaseModel):
    dataset_path: str = "evaluation/datasets/golden_qa.jsonl"
    reports_dir: str = "evaluation/reports"

    # retrieval
    k_values: list[int] = [1, 3, 5]
    min_recall_at_5: float = 0.7  # CI gate

    # generation (LLM-as-judge; judge model comes from LLM__* env like the app,
    # but SHOULD be pointed at a different/stronger model than the answerer)
    min_faithfulness: float = 0.7
    min_citation_accuracy: float = 0.8
