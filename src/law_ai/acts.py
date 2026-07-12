"""Registry of legal acts covered by the offline ingestion pipeline.

Static domain knowledge, not deployment config: each act points to the ISAP
consolidated-text PDF (the "…Lj.pdf" links, always the current unified text).
The ingestion DAG factory builds one DAG per domain from this registry —
expanding coverage means adding entries here, nothing else.
"""

from pydantic import BaseModel


class LegalAct(BaseModel):
    act_id: str  # slug: chunk-id prefix + `metadata.act` filter value
    name: str  # official short title (Polish)
    domain: str  # legal domain slug, e.g. "civil" — `metadata.domain` filter value
    url: str  # ISAP consolidated-text PDF
    effective_date: str = ""  # ISO date the act entered into force


ACTS: list[LegalAct] = [
    LegalAct(
        act_id="konstytucja",
        name="Konstytucja Rzeczypospolitej Polskiej",
        domain="constitutional",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19970780483/U/D19970483Lj.pdf",
        effective_date="1997-10-17",
    ),
    # ------------------------------------------------------------- civil ----
    LegalAct(
        act_id="kodeks-cywilny",
        name="Kodeks cywilny",
        domain="civil",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19640160093/U/D19640093Lj.pdf",
        effective_date="1965-01-01",
    ),
    LegalAct(
        act_id="ochrona-praw-lokatorow",
        name="Ustawa o ochronie praw lokatorów, mieszkaniowym zasobie gminy "
        "i o zmianie Kodeksu cywilnego",
        domain="civil",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20010710733/U/D20010733Lj.pdf",
        effective_date="2001-07-10",
    ),
    LegalAct(
        act_id="wlasnosc-lokali",
        name="Ustawa o własności lokali",
        domain="civil",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19940850388/U/D19940388Lj.pdf",
        effective_date="1995-01-01",
    ),
    LegalAct(
        act_id="ksiegi-wieczyste-hipoteka",
        name="Ustawa o księgach wieczystych i hipotece",
        domain="civil",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19820190147/U/D19820147Lj.pdf",
        effective_date="1983-01-01",
    ),
]


def domains() -> list[str]:
    """Distinct domains in registry order."""
    seen: dict[str, None] = {}
    for act in ACTS:
        seen.setdefault(act.domain, None)
    return list(seen)


def acts_for_domain(domain: str) -> list[LegalAct]:
    return [act for act in ACTS if act.domain == domain]


def get_act(act_id: str) -> LegalAct:
    for act in ACTS:
        if act.act_id == act_id:
            return act
    raise KeyError(f"Unknown act_id: {act_id!r}")
