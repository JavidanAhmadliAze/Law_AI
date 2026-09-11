"""Query planner — deterministic. Rewrite, decompose and translate in ONE call.

This node used to be followed by a separate `translator` node. Merging them
removed the single largest avoidable cost in the chain: translating one sentence
was measured at 40.9s mean (111s worst case) across a golden run — 31% of total
latency for a round trip that produced ~30 tokens.

The merge is safe because both steps are pure functions of the same input: what
must be researched, and how to say it in Polish. Neither needs the other's output
as a separate model turn.

The glossary is preserved, but as a terminology table in the prompt rather than
a regex substitution on the question. Substitution was context-blind — "the
president of a company" became "Prezydent", the head of state — and fed the
model half-Polish input. The table keeps the canonical wording reaching the BM25
leg while leaving the model to judge whether a concept actually applies.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from law_ai.services.agents import prompt
from law_ai.services.agents.context import last_human, services_from_config
from law_ai.services.agents.schema import RewrittenQuery
from law_ai.services.agents.state import AgentOutputState


async def query_rewriter(state: AgentOutputState, config: RunnableConfig) -> dict[str, Any]:
    services = services_from_config(config)
    question = last_human(list(state["messages"]))
    rewritten: RewrittenQuery = await services.fast_llm.generate_structured(
        prompt.query_rewriter_prompt(), question, RewrittenQuery
    )
    # How many searches a question needs is a reading of the question, so it is
    # settled here: sub-queries when the issues are independent, otherwise the
    # brief as a single search. Downstream always receives a non-empty list and
    # never has to reason about decomposition again.
    #
    # Not truncated: a question with five issues needs five searches. That cost
    # is bounded by the retriever's rerank budget, not by discarding issues the
    # rewriter correctly identified.
    return {
        "research_brief_pl": rewritten.research_brief_pl,
        "search_queries_pl": rewritten.sub_queries_pl or [rewritten.research_brief_pl],
    }
