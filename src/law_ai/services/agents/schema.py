"""Structured outputs for the graph's LLM calls.

The guardian returns a verdict; the planning subgraph returns an Analysis and
then a Plan. Field order is generation order: the model fills `questions` before
`institutions`, and an institution's `evidence` before the acts that govern it,
so each field is decided with the previous ones already written.

Nothing here describes retrieval. The Plan's points become searches through
`prompt.to_queries`, which is checked code, not a model output.
"""

from pydantic import BaseModel, Field
from typing import Dict
from typing import Literal


class GuardianVerdict(BaseModel):
    allowed: bool = Field(description="True if the question may enter the pipeline")
    reason: str = Field(description="One of: ok | off_topic | injection | unsafe")
    message: str = Field(default="", description="User-facing refusal text when blocked")

class QueryPlan(BaseModel):

    plan: Dict[str, str] = Field(
        description="A dictionary mapping specific sub-queries to their corresponding act."
    )




ACTS = Literal[
    "akcyza",
    "cit",
    "doradztwo-podatkowe",
    "egzekucja-administracyjna",
    "kas",
    "kodeks-cywilny",
    "kodeks-karny-skarbowy",
    "kodeks-postepowania-administracyjnego",
    "kodeks-pracy",
    "konstytucja",
    "ochrona-praw-lokatorow",
    "ordynacja-podatkowa",
    "pcc",
    "pit",
    "podatek-lesny",
    "podatek-rolny",
    "podatki-oplaty-lokalne",
    "prawo-celne",
    "prawo-przedsiebiorcow",
    "ryczalt-ewidencjonowany",
    "sady-administracyjne",
    "spadki-darowizny",
    "sprzedaz-detaliczna",
    "ubezpieczenia-spoleczne",
    "vat",
    "ceidg",
    "finanse-publiczne",
    "kodeks-spolek-handlowych",
    "krajowy-rejestr-sadowy",
    "ksiegi-wieczyste-hipoteka",
    "nieuczciwa-konkurencja",
    "przedsiebiorcy-zagraniczni",
    "rachunkowosc",
    "wlasnosc-lokali",
]
