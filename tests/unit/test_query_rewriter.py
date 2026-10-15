"""The planning node: the model's plan becomes the retriever's searches
through checked code — cleaned, act-validated, capped, never empty."""

from typing import Any, cast

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from law_ai.services.agents.context import AgentServices
from law_ai.services.agents.nodes.query_rewriter import _clean, _load_bank, query_rewriter
from law_ai.services.agents.schema import QueryPlan


class _FakeLLM:
    def __init__(self, plan: dict[str, str]) -> None:
        self.plan, self.calls = plan, []

    async def generate_structured(self, system: str, user: str, schema: Any) -> Any:
        self.calls.append((system, user))
        return QueryPlan(plan=self.plan)


def _config(llm: Any, **kw: Any) -> RunnableConfig:
    services = AgentServices(llm=cast(Any, llm), search=cast(Any, object()), max_searches=3, **kw)
    return cast(RunnableConfig, {"configurable": {"services": services}})


def _state(question: str) -> Any:
    return {"messages": [HumanMessage(content=question)]}


def test_clean_normalises_dedupes_validates_acts_and_caps() -> None:
    plan = {
        " obowiązek  zapłaty akcyzy ": "akcyza",
        "Obowiązek zapłaty akcyzy": "akcyza",  # duplicate, case-insensitive
        "przedawnienie zobowiązania podatkowego.": "ordynacja-podatkowa",
        "zwrot nadpłaty": "nie-ma-takiego-aktu",  # unknown act -> unscoped
        "piąta etykieta": "pit",
    }
    assert _clean(plan, max_searches=3) == {
        "obowiązek zapłaty akcyzy": "akcyza",
        "przedawnienie zobowiązania podatkowego": "ordynacja-podatkowa",
        "zwrot nadpłaty": "",
    }


@pytest.mark.asyncio
async def test_node_returns_the_cleaned_plan_and_uses_the_reasoning_llm() -> None:
    chat, reasoning = _FakeLLM({"ignored": "pit"}), _FakeLLM({"stawka podatku od umowy sprzedaży": "pcc"})
    out = await query_rewriter(_state("Kupiłem auto"), _config(chat, reasoning_llm=cast(Any, reasoning)))
    assert out == {"search_queries_pl": {"stawka podatku od umowy sprzedaży": "pcc"}}
    assert reasoning.calls and not chat.calls
    system, user = reasoning.calls[0]
    assert "KATALOG AKTÓW" in system and "- pcc:" in system and user.endswith("Kupiłem auto")


@pytest.mark.asyncio
async def test_empty_plan_falls_back_to_the_question_over_the_whole_corpus() -> None:
    out = await query_rewriter(_state("Co to jest?"), _config(_FakeLLM({})))
    assert out == {"search_queries_pl": {"Co to jest?": ""}}


@pytest.mark.asyncio
async def test_label_bank_is_shown_grouped_by_act(tmp_path: Any) -> None:
    bank = tmp_path / "bank.jsonl"
    bank.write_text(
        '{"id": 1, "label": {"zajęcie wierzytelności z rachunku bankowego": "egzekucja-administracyjna"}}\n'
        '{"id": 2, "label": {"nieznany akt": "x"}}\n',
        encoding="utf-8",
    )
    _load_bank.cache_clear()
    llm = _FakeLLM({"zajęcie wierzytelności z rachunku bankowego": "egzekucja-administracyjna"})
    await query_rewriter(_state("Komornik zajął konto"), _config(llm, label_bank_path=str(bank)))
    system = llm.calls[0][0]
    assert "BANK ETYKIET" in system and "[egzekucja-administracyjna]" in system
    assert "nieznany akt" not in system
