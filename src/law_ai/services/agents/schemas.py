"""Structured outputs for every LLM call in the agent graph.

Typed results make graph routing deterministic and each node testable —
no free-text parsing anywhere.
"""

from pydantic import BaseModel, Field


class GuardianVerdict(BaseModel):
    allowed: bool = Field(description="True if the question may enter the pipeline")
    reason: str = Field(
        description="One of: ok | off_topic | injection | unsafe",
    )
    message: str = Field(default="", description="User-facing refusal text when blocked")


class RewrittenQuery(BaseModel):
    route: str = Field(description="'simple' for one sub-question, 'complex' for several")
    sub_questions: list[str] = Field(
        description="Self-contained sub-questions, in the user's language"
    )
    article_filter: str = Field(
        default="", description="Article reference if the user asked about one, e.g. 'Art. 54'"
    )


class ThinkResult(BaseModel):
    sufficient: bool = Field(description="Does the evidence fully answer the question?")
    missing: str = Field(default="", description="What is missing, if anything")
    refined_query: str = Field(
        default="", description="A better retrieval query to fill the gap, if needed"
    )


class EvidenceItem(BaseModel):
    claim: str = Field(description="A fact that helps answer the question, in English")
    source_article: str = Field(description="e.g. 'Art. 54'")
    quote: str = Field(description="Verbatim Polish quote from the source chunk — never translated")


class CompressedEvidence(BaseModel):
    items: list[EvidenceItem]


class SubAgentResult(BaseModel):
    sub_question: str
    evidence: list[EvidenceItem] = []
    sufficient: bool = False


class SupervisorDecision(BaseModel):
    complete: bool = Field(description="True when coverage is good enough to write the answer")
    additional_questions: list[str] = Field(
        default_factory=list,
        description="New sub-questions to dispatch if coverage is incomplete",
    )


class Citation(BaseModel):
    article: str
    quote: str


class FinalAnswer(BaseModel):
    answer: str = Field(description="Grounded answer in the user's language")
    citations: list[Citation] = Field(default_factory=list)
