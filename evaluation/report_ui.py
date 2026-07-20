"""Render evaluation reports into a self-contained HTML dashboard.

    uv run python -m evaluation.report_ui

Reads every JSON report in evaluation/reports/ (both the Langfuse-experiment
format {summary, per_item} and run_eval's {summary, per_query}) and writes
evaluation/reports/eval_dashboard.html — no server, no CDN, open in a browser.
Re-run after each evaluation to refresh.
"""

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation.config import EvalConfig

_METRICS = ("hit@5", "recall@5", "precision@5", "mrr", "ndcg@5")


def _load_runs(reports_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        summary = data.get("summary")
        rows = data.get("per_item") or data.get("per_query")
        if not summary or not rows:
            continue
        runs.append(
            {
                "name": summary.get("run_name") or path.stem,
                "file": path.name,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
                "n": len(rows),
                "metrics": {m: summary.get(m) for m in _METRICS if summary.get(m) is not None},
                "rows": rows,
            }
        )
    return runs


def _act_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_act: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        expected = row.get("expected") or []
        act = expected[0].split(":")[0] if expected else "unknown"
        by_act[act].append(row)
    return sorted(
        (
            {
                "act": act,
                "n": len(items),
                "hit5": round(mean(r.get("hit@5", 0.0) for r in items), 4),
                "mrr": round(mean(r.get("mrr", 0.0) for r in items), 4),
            }
            for act, items in by_act.items()
        ),
        key=lambda a: -a["hit5"],
    )


def render(runs: list[dict[str, Any]]) -> str:
    latest = runs[-1]
    payload = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "latest": {k: latest[k] for k in ("name", "file", "n", "metrics", "rows")},
        "acts": _act_stats(latest["rows"]),
        "history": [
            {
                "name": r["name"],
                "date": r["mtime"].strftime("%Y-%m-%d %H:%M"),
                "n": r["n"],
                "metrics": r["metrics"],
            }
            for r in runs
        ],
    }
    template = Path(__file__).with_name("report_template.html").read_text()
    return template.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))


def main() -> None:
    reports_dir = Path(EvalConfig().reports_dir)
    runs = _load_runs(reports_dir)
    if not runs:
        raise SystemExit(f"no usable reports in {reports_dir}/")
    out = reports_dir / "eval_dashboard.html"
    out.write_text(render(runs))
    print(f"dashboard: {out}  ({len(runs)} runs, latest: {runs[-1]['name']})")


if __name__ == "__main__":
    main()
