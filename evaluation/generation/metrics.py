"""LLM-as-judge metrics for generation quality.

The judge model comes from LLM__* env (point it at a stronger/different model
than the answerer when running seriously). Citation accuracy is checked
deterministically against expected articles — no judge needed for that.
"""

from pydantic import BaseModel, Field

from law_ai.services.llm.base import BaseLLM

_JUDGE_PROMPT = """\
You are a strict evaluator of a legal RAG system over the Polish Constitution.
Given a question, a reference answer, and the system's answer, score:
- faithfulness (0-1): does the system answer avoid inventing facts not in the reference/evidence?
- relevance (0-1): does it actually address the question?
- correctness (0-1): does it agree with the reference answer?
Be harsh: fabricated article numbers or invented rights mean faithfulness <= 0.2."""


class JudgeScores(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    correctness: float = Field(ge=0, le=1)
    comment: str = ""


async def judge_answer(llm: BaseLLM, *, question: str, reference: str, answer: str) -> JudgeScores:
    return await llm.generate_structured(
        _JUDGE_PROMPT,
        f"Question: {question}\n\nReference answer: {reference}\n\nSystem answer: {answer}",
        JudgeScores,
    )


def citation_accuracy(cited_articles: list[str], expected_articles: list[str]) -> float:
    """Fraction of citations that point at expected articles (deterministic)."""
    if not cited_articles:
        return 0.0
    expected = set(expected_articles)
    return sum(1 for a in cited_articles if a in expected) / len(cited_articles)
