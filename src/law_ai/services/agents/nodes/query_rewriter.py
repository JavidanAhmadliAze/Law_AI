"""Query rewriter — the planning step, one LLM call.

Turns the user's question into {sub-query: act} pairs for the retriever. A
sub-query is the label of ONE provision in the statute's own wording — the
form a lawyer uses when citing — and the act is the one it is confined to
("" when the model cannot place it, which searches the whole corpus).

Measured on held-out questions (subquery_eval/): a reasoning model picks the
lawyer's provision markedly more often than the chat model, and a bank of
lawyer-written labels shown grouped by act adds another 9-11 points of
recall. So the node uses the reasoning LLM when one is configured and the
label bank when `AGENT__LABEL_BANK_PATH` points at one. Nothing here
retrieves: the retriever is the only place the corpus is read.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from law_ai.acts import ACTS
from law_ai.services.agents import prompt
from law_ai.services.agents.context import last_human, services_from_config
from law_ai.services.agents.schema import QueryPlan
from law_ai.services.agents.state import AgentState

ACT_IDS = {act.act_id for act in ACTS}
CATALOG = "\n".join(f"- {act.act_id}: {act.name}" for act in ACTS)


@lru_cache(maxsize=4)
def _load_bank(path: str) -> dict[str, str]:
    """Lawyer labels from a JSONL of rows with a `label` {text: act} dict.
    Cached per path: the file is read once per process, not per question."""
    if not path:
        return {}
    bank: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for text, act in (json.loads(line).get("label") or {}).items():
            if act in ACT_IDS:
                bank.setdefault(" ".join(text.split()), act)
    return bank


def _clean(plan: dict[str, str], max_searches: int) -> dict[str, str]:
    """Normalise whitespace, drop empties and duplicates, keep only catalogue
    acts ("" = unscoped), cap the count — checked code, not a model output."""
    searches: dict[str, str] = {}
    for text, act in plan.items():
        query = " ".join(str(text).split()).strip(" .;:,")
        if not query or query.casefold() in {q.casefold() for q in searches}:
            continue
        searches[query] = act if act in ACT_IDS else ""
        if len(searches) >= max_searches:
            break
    return searches


async def query_rewriter(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    services = services_from_config(config)
    question = last_human(list(state["messages"]))
    if not services.decompose_queries:
        return {"search_queries_pl": {question: ""}}
    llm = services.reasoning_llm or services.llm
    if llm is None:
        raise RuntimeError("query_rewriter needs an LLM in the agent services")

    system = prompt.planner_system(
        catalog=CATALOG,
        bank=_load_bank(services.label_bank_path),
        max_labels=services.max_searches or 4,
    )
    plan = await llm.generate_structured(system, f"PYTANIE:\n{question}", QueryPlan)
    searches = _clean(plan.plan, services.max_searches or 4)
    # The retriever must always have something to search: a plan the model
    # could not make falls back to the question itself over the whole corpus.
    return {"search_queries_pl": searches or {question: ""}}
