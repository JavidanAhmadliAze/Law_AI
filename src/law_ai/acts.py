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
    # ---------------------------------------------------------- business ----
    LegalAct(
        act_id="kodeks-spolek-handlowych",
        name="Kodeks spółek handlowych",
        domain="business",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20000941037/U/D20001037Lj.pdf",
        effective_date="2001-01-01",
    ),
    LegalAct(
        act_id="prawo-przedsiebiorcow",
        name="Prawo przedsiębiorców",
        domain="business",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20180000646/U/D20180646Lj.pdf",
        effective_date="2018-04-30",
    ),
    LegalAct(
        act_id="przedsiebiorcy-zagraniczni",
        name="Ustawa o zasadach uczestnictwa przedsiębiorców zagranicznych i innych osób "
        "zagranicznych w obrocie gospodarczym na terytorium RP",
        domain="business",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20180000649/U/D20180649Lj.pdf",
        effective_date="2018-04-30",
    ),
    LegalAct(
        act_id="krajowy-rejestr-sadowy",
        name="Ustawa o Krajowym Rejestrze Sądowym",
        domain="business",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19971210769/U/D19970769Lj.pdf",
        effective_date="2001-01-01",
    ),
    LegalAct(
        act_id="ceidg",
        name="Ustawa o Centralnej Ewidencji i Informacji o Działalności Gospodarczej "
        "i Punkcie Informacji dla Przedsiębiorcy",
        domain="business",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20180000647/U/D20180647Lj.pdf",
        effective_date="2018-04-30",
    ),
    LegalAct(
        act_id="rachunkowosc",
        name="Ustawa o rachunkowości",
        domain="business",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19941210591/U/D19940591Lj.pdf",
        effective_date="1995-01-01",
    ),
    LegalAct(
        act_id="nieuczciwa-konkurencja",
        name="Ustawa o zwalczaniu nieuczciwej konkurencji",
        domain="business",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19930470211/U/D19930211Lj.pdf",
        effective_date="1993-08-16",
    ),
    # -------------------------------------------------------- employment ----
    LegalAct(
        act_id="kodeks-pracy",
        name="Kodeks pracy",
        domain="employment",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19740240141/U/D19740141Lj.pdf",
        effective_date="1975-01-01",
    ),
    LegalAct(
        act_id="ubezpieczenia-spoleczne",
        name="Ustawa o systemie ubezpieczeń społecznych",
        domain="employment",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19981370887/U/D19980887Lj.pdf",
        effective_date="1999-01-01",
    ),
    # --------------------------------------------------------------- tax ----
    LegalAct(
        act_id="vat",
        name="Ustawa o podatku od towarów i usług (VAT)",
        domain="tax",
        url="https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20040540535/U/D20040535Lj.pdf",
        effective_date="2004-05-01",
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
