"""Registry of legal acts covered by the offline ingestion pipeline.

Static domain knowledge, not deployment config: each act points to the Sejm ELI
API's consolidated-text PDF (the "…Lj.pdf" unified text, always current).

The API is used rather than the ISAP download page because ISAP sits behind bot
protection that answers HTTP 200 with an HTML challenge instead of the PDF —
the fetcher cannot tell that apart from a real download without inspecting the
body, and retries do not clear it. api.sejm.gov.pl serves the identical files
with no challenge.
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
        url="https://api.sejm.gov.pl/eli/acts/DU/1997/483/text/U/D19970483Lj.pdf",
        effective_date="1997-10-17",
    ),
    # ------------------------------------------------------------- civil ----
    LegalAct(
        act_id="kodeks-cywilny",
        name="Kodeks cywilny",
        domain="civil",
        url="https://api.sejm.gov.pl/eli/acts/DU/1964/93/text/U/D19640093Lj.pdf",
        effective_date="1965-01-01",
    ),
    LegalAct(
        act_id="ochrona-praw-lokatorow",
        name="Ustawa o ochronie praw lokatorów, mieszkaniowym zasobie gminy "
        "i o zmianie Kodeksu cywilnego",
        domain="civil",
        url="https://api.sejm.gov.pl/eli/acts/DU/2001/733/text/U/D20010733Lj.pdf",
        effective_date="2001-07-10",
    ),
    LegalAct(
        act_id="wlasnosc-lokali",
        name="Ustawa o własności lokali",
        domain="civil",
        url="https://api.sejm.gov.pl/eli/acts/DU/1994/388/text/U/D19940388Lj.pdf",
        effective_date="1995-01-01",
    ),
    LegalAct(
        act_id="ksiegi-wieczyste-hipoteka",
        name="Ustawa o księgach wieczystych i hipotece",
        domain="civil",
        url="https://api.sejm.gov.pl/eli/acts/DU/1982/147/text/U/D19820147Lj.pdf",
        effective_date="1983-01-01",
    ),
    # ---------------------------------------------------------- business ----
    LegalAct(
        act_id="kodeks-spolek-handlowych",
        name="Kodeks spółek handlowych",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/2000/1037/text/U/D20001037Lj.pdf",
        effective_date="2001-01-01",
    ),
    LegalAct(
        act_id="prawo-przedsiebiorcow",
        name="Prawo przedsiębiorców",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/2018/646/text/U/D20180646Lj.pdf",
        effective_date="2018-04-30",
    ),
    LegalAct(
        act_id="przedsiebiorcy-zagraniczni",
        name="Ustawa o zasadach uczestnictwa przedsiębiorców zagranicznych i innych osób "
        "zagranicznych w obrocie gospodarczym na terytorium RP",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/2018/649/text/U/D20180649Lj.pdf",
        effective_date="2018-04-30",
    ),
    LegalAct(
        act_id="krajowy-rejestr-sadowy",
        name="Ustawa o Krajowym Rejestrze Sądowym",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/1997/769/text/U/D19970769Lj.pdf",
        effective_date="2001-01-01",
    ),
    LegalAct(
        act_id="ceidg",
        name="Ustawa o Centralnej Ewidencji i Informacji o Działalności Gospodarczej "
        "i Punkcie Informacji dla Przedsiębiorcy",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/2018/647/text/U/D20180647Lj.pdf",
        effective_date="2018-04-30",
    ),
    LegalAct(
        act_id="rachunkowosc",
        name="Ustawa o rachunkowości",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/1994/591/text/U/D19940591Lj.pdf",
        effective_date="1995-01-01",
    ),
    LegalAct(
        act_id="nieuczciwa-konkurencja",
        name="Ustawa o zwalczaniu nieuczciwej konkurencji",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/1993/211/text/U/D19930211Lj.pdf",
        effective_date="1993-08-16",
    ),
    # -------------------------------------------------------- employment ----
    LegalAct(
        act_id="kodeks-pracy",
        name="Kodeks pracy",
        domain="employment",
        url="https://api.sejm.gov.pl/eli/acts/DU/1974/141/text/U/D19740141Lj.pdf",
        effective_date="1975-01-01",
    ),
    LegalAct(
        act_id="ubezpieczenia-spoleczne",
        name="Ustawa o systemie ubezpieczeń społecznych",
        domain="employment",
        url="https://api.sejm.gov.pl/eli/acts/DU/1998/887/text/U/D19980887Lj.pdf",
        effective_date="1999-01-01",
    ),
    # --------------------------------------------------------------- tax ----
    LegalAct(
        act_id="vat",
        name="Ustawa o podatku od towarów i usług (VAT)",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/2004/535/text/U/D20040535Lj.pdf",
        effective_date="2004-05-01",
    ),
    # ------------------------------------------------------------- civil ----
    LegalAct(
        act_id="kodeks-postepowania-cywilnego",
        name="Ustawa z dnia 17 listopada 1964 r. - Kodeks postępowania cywilnego.",
        domain="civil",
        url="https://api.sejm.gov.pl/eli/acts/DU/1964/296/text/U/D19640296Lj.pdf",
        effective_date="1965-01-01",
    ),
    LegalAct(
        act_id="kodeks-rodzinny-opiekunczy",
        name="Ustawa z dnia 25 lutego 1964 r. - Kodeks rodzinny i opiekuńczy.",
        domain="civil",
        url="https://api.sejm.gov.pl/eli/acts/DU/1964/59/text/U/D19640059Lj.pdf",
        effective_date="1965-01-01",
    ),
    LegalAct(
        act_id="prawa-konsumenta",
        name="Ustawa z dnia 30 maja 2014 r. o prawach konsumenta",
        domain="civil",
        url="https://api.sejm.gov.pl/eli/acts/DU/2014/827/text/U/D20140827Lj.pdf",
        effective_date="2014-12-25",
    ),
    LegalAct(
        act_id="prawo-autorskie",
        name="Ustawa z dnia 4 lutego 1994 r. o prawie autorskim i prawach pokrewnych.",
        domain="civil",
        url="https://api.sejm.gov.pl/eli/acts/DU/1994/83/text/U/D19940083Lj.pdf",
        effective_date="1994-05-23",
    ),
    # ---------------------------------------------------------- business ----
    LegalAct(
        act_id="prawo-bankowe",
        name="Ustawa z dnia 29 sierpnia 1997 r. - Prawo bankowe.",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/1997/939/text/U/D19970939Lj.pdf",
        effective_date="1998-01-01",
    ),
    LegalAct(
        act_id="prawo-upadlosciowe",
        name="Ustawa z dnia 28 lutego 2003 r. Prawo upadłościowe",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/2003/535/text/U/D20030535Lj.pdf",
        effective_date="2003-10-01",
    ),
    LegalAct(
        act_id="prawo-restrukturyzacyjne",
        name="Ustawa z dnia 15 maja 2015 r. - Prawo restrukturyzacyjne",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/2015/978/text/U/D20150978Lj.pdf",
        effective_date="2016-01-01",
    ),
    LegalAct(
        act_id="zamowienia-publiczne",
        name="Ustawa z dnia 11 września 2019 r. - Prawo zamówień publicznych",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/2019/2019/text/U/D20192019Lj.pdf",
        effective_date="2021-01-01",
    ),
    # -------------------------------------------------------- employment ----
    LegalAct(
        act_id="swiadczenia-chorobowe",
        name="Ustawa z dnia 25 czerwca 1999 r. o świadczeniach pieniężnych z ubezpieczenia społecznego w razie choroby i macierzyństwa.",
        domain="employment",
        url="https://api.sejm.gov.pl/eli/acts/DU/1999/636/text/U/D19990636Lj.pdf",
        effective_date="1999-09-01",
    ),
    # --------------------------------------------------------------- tax ----
    LegalAct(
        act_id="ordynacja-podatkowa",
        name="Ustawa z dnia 29 sierpnia 1997 r. - Ordynacja podatkowa.",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/1997/926/text/U/D19970926Lj.pdf",
        effective_date="1998-01-01",
    ),
    LegalAct(
        act_id="pit",
        name="Ustawa z dnia 26 lipca 1991 r. o podatku dochodowym od osób fizycznych.",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/1991/350/text/U/D19910350Lj.pdf",
        effective_date="1992-01-01",
    ),
    LegalAct(
        act_id="cit",
        name="Ustawa z dnia 15 lutego 1992 r. o podatku dochodowym od osób prawnych",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/1992/86/text/U/D19920086Lj.pdf",
        effective_date="1992-03-10",
    ),
    LegalAct(
        act_id="akcyza",
        name="Ustawa z dnia 6 grudnia 2008 r. o podatku akcyzowym",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/2009/11/text/U/D20090011Lj.pdf",
        effective_date="2009-03-01",
    ),
    LegalAct(
        act_id="finanse-publiczne",
        name="Ustawa z dnia 27 sierpnia 2009 r. o finansach publicznych",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/2009/1240/text/U/D20091240Lj.pdf",
        effective_date="2010-01-01",
    ),
    LegalAct(
        act_id="podatki-oplaty-lokalne",
        name="Ustawa z dnia 12 stycznia 1991 r. o podatkach i opłatach lokalnych.",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/1991/31/text/U/D19910031Lj.pdf",
        effective_date="1991-01-30",
    ),
    # ---------------------------------------------------------- criminal ----
    LegalAct(
        act_id="kodeks-karny",
        name="Ustawa z dnia 6 czerwca 1997 r. - Kodeks karny.",
        domain="criminal",
        url="https://api.sejm.gov.pl/eli/acts/DU/1997/553/text/U/D19970553Lj.pdf",
        effective_date="1998-09-01",
    ),
    LegalAct(
        act_id="kodeks-postepowania-karnego",
        name="Ustawa z dnia 6 czerwca 1997 r. - Kodeks postępowania karnego.",
        domain="criminal",
        url="https://api.sejm.gov.pl/eli/acts/DU/1997/555/text/U/D19970555Lj.pdf",
        effective_date="1998-09-01",
    ),
    LegalAct(
        act_id="kodeks-karny-wykonawczy",
        name="Ustawa z dnia 6 czerwca 1997 r. - Kodeks karny wykonawczy.",
        domain="criminal",
        url="https://api.sejm.gov.pl/eli/acts/DU/1997/557/text/U/D19970557Lj.pdf",
        effective_date="1998-09-01",
    ),
    LegalAct(
        act_id="kodeks-wykroczen",
        name="Ustawa z dnia 20 maja 1971 r. Kodeks wykroczeń.",
        domain="criminal",
        url="https://api.sejm.gov.pl/eli/acts/DU/1971/114/text/U/D19710114Lj.pdf",
        effective_date="1972-01-01",
    ),
    LegalAct(
        act_id="kodeks-karny-skarbowy",
        name="Ustawa z dnia 10 września 1999 r. Kodeks karny skarbowy.",
        domain="criminal",
        url="https://api.sejm.gov.pl/eli/acts/DU/1999/930/text/U/D19990930Lj.pdf",
        effective_date="1999-10-17",
    ),
    LegalAct(
        act_id="postepowanie-wykroczenia",
        name="Ustawa z dnia 24 sierpnia 2001 r. - Kodeks postępowania w sprawach o wykroczenia.",
        domain="criminal",
        url="https://api.sejm.gov.pl/eli/acts/DU/2001/1148/text/U/D20011148Lj.pdf",
        effective_date="2001-10-17",
    ),
    # ---------------------------------------------------- administrative ----
    LegalAct(
        act_id="kodeks-postepowania-administracyjnego",
        name="Ustawa z dnia 14 czerwca 1960 r. Kodeks postępowania administracyjnego.",
        domain="administrative",
        url="https://api.sejm.gov.pl/eli/acts/DU/1960/168/text/U/D19600168Lj.pdf",
        effective_date="1961-01-01",
    ),
    LegalAct(
        act_id="sady-administracyjne",
        name="Ustawa z dnia 30 sierpnia 2002 r. Prawo o postępowaniu przed sądami administracyjnymi.",
        domain="administrative",
        url="https://api.sejm.gov.pl/eli/acts/DU/2002/1270/text/U/D20021270Lj.pdf",
        effective_date="2004-01-01",
    ),
    LegalAct(
        act_id="samorzad-gminny",
        name="Ustawa z dnia 8 marca 1990 r. o samorządzie gminnym",
        domain="administrative",
        url="https://api.sejm.gov.pl/eli/acts/DU/1990/95/text/U/D19900095Lj.pdf",
        effective_date="1990-05-27",
    ),
    LegalAct(
        act_id="informacja-publiczna",
        name="Ustawa z dnia 6 września 2001 r. o dostępie do informacji publicznej.",
        domain="administrative",
        url="https://api.sejm.gov.pl/eli/acts/DU/2001/1198/text/U/D20011198Lj.pdf",
        effective_date="2002-01-01",
    ),
    LegalAct(
        act_id="ochrona-danych-osobowych",
        name="Ustawa z dnia 10 maja 2018 r. o ochronie danych osobowych",
        domain="administrative",
        url="https://api.sejm.gov.pl/eli/acts/DU/2018/1000/text/U/D20181000Lj.pdf",
        effective_date="2018-05-25",
    ),
    # ------------------------------------------------------- codes added ----
    # The remaining statutory codes. The EU Customs Code (Reg. 952/2013) has no
    # entry: it is EU law published in the Official Journal, never in Dziennik
    # Ustaw, so the ELI API does not serve it — `prawo-celne` is the Polish
    # national customs act that complements it.
    LegalAct(
        act_id="kodeks-morski",
        name="Kodeks morski",
        domain="business",
        url="https://api.sejm.gov.pl/eli/acts/DU/2001/1545/text/U/D20011545Lj.pdf",
        effective_date="2002-06-05",
    ),
    LegalAct(
        act_id="kodeks-wyborczy",
        name="Kodeks wyborczy",
        domain="constitutional",
        url="https://api.sejm.gov.pl/eli/acts/DU/2011/112/text/U/D20110112Lj.pdf",
        effective_date="2011-08-01",
    ),
    LegalAct(
        act_id="prawo-celne",
        name="Prawo celne",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/2004/622/text/U/D20040622Lj.pdf",
        effective_date="2004-05-01",
    ),
    # ---- the rest of the tax system: the substantive taxes, the authority
    # that administers them and the procedure that enforces them. Without
    # these the domain answers on VAT/PIT/CIT and nothing else — a freelancer
    # on ryczalt, a flat purchase (PCC), an inheritance or a seized account
    # all fall outside the four acts originally registered here.
    LegalAct(
        act_id="ryczalt-ewidencjonowany",
        name="Ustawa o zryczałtowanym podatku dochodowym od niektórych przychodów "
        "osiąganych przez osoby fizyczne",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/1998/930/text/U/D19980930Lj.pdf",
        effective_date="1999-01-01",
    ),
    LegalAct(
        act_id="pcc",
        name="Ustawa o podatku od czynności cywilnoprawnych",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/2000/959/text/U/D20000959Lj.pdf",
        effective_date="2001-01-01",
    ),
    LegalAct(
        act_id="spadki-darowizny",
        name="Ustawa o podatku od spadków i darowizn",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/1983/207/text/U/D19830207Lj.pdf",
        effective_date="1983-09-01",
    ),
    LegalAct(
        act_id="podatek-rolny",
        name="Ustawa o podatku rolnym",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/1984/268/text/U/D19840268Lj.pdf",
        effective_date="1985-01-01",
    ),
    LegalAct(
        act_id="podatek-lesny",
        name="Ustawa o podatku leśnym",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/2002/1682/text/U/D20021682Lj.pdf",
        effective_date="2003-01-01",
    ),
    LegalAct(
        act_id="kas",
        name="Ustawa o Krajowej Administracji Skarbowej",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/2016/1947/text/U/D20161947Lj.pdf",
        effective_date="2017-03-01",
    ),
    LegalAct(
        act_id="egzekucja-administracyjna",
        name="Ustawa o postępowaniu egzekucyjnym w administracji",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/1966/151/text/U/D19660151Lj.pdf",
        effective_date="1967-01-01",
    ),
    LegalAct(
        act_id="doradztwo-podatkowe",
        name="Ustawa o doradztwie podatkowym",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/1996/475/text/U/D19960475Lj.pdf",
        effective_date="1997-01-01",
    ),
    LegalAct(
        act_id="sprzedaz-detaliczna",
        name="Ustawa o podatku od sprzedaży detalicznej",
        domain="tax",
        url="https://api.sejm.gov.pl/eli/acts/DU/2016/1155/text/U/D20161155Lj.pdf",
        effective_date="2016-09-01",
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
