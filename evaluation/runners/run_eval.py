"""Evaluation runner.

    uv run python -m evaluation.runners.run_eval --mode retrieval
    uv run python -m evaluation.runners.run_eval --mode generation

Writes a JSON report to evaluation/reports/ and exits non-zero when a CI
threshold from EvalConfig is violated.
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from evaluation.config import EvalConfig


async def main(mode: str) -> int:
    config = EvalConfig()
    if mode == "retrieval":
        from evaluation.retrieval.evaluate import evaluate_retrieval

        report = await evaluate_retrieval(config)
        gate_ok = report["summary"].get("recall@5", 0.0) >= config.min_recall_at_5
        gate_msg = f"recall@5 >= {config.min_recall_at_5}"
    elif mode == "generation":
        from evaluation.generation.evaluate import evaluate_generation

        report = await evaluate_generation(config)
        summary = report["summary"]
        gate_ok = (
            summary.get("faithfulness", 0.0) >= config.min_faithfulness
            and summary.get("citation_accuracy", 0.0) >= config.min_citation_accuracy
        )
        gate_msg = (
            f"faithfulness >= {config.min_faithfulness} "
            f"and citation_accuracy >= {config.min_citation_accuracy}"
        )
    else:
        print(f"unknown mode: {mode}", file=sys.stderr)
        return 2

    reports_dir = Path(config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"{mode}-{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"report: {report_path}")
    if not gate_ok:
        print(f"❌ threshold violated: {gate_msg}", file=sys.stderr)
        return 1
    print(f"✅ thresholds OK ({gate_msg})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["retrieval", "generation"], required=True)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.mode)))
