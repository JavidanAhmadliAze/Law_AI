"""Deterministic EN→PL legal-terminology glossary.

Regex-based, trusted mappings applied BEFORE any model/LLM translation so
canonical Polish legal terms reach the BM25 leg verbatim. Extend freely —
each entry is (case-insensitive English phrase → canonical Polish term).
"""

import re

# fmt: off
LEGAL_GLOSSARY: dict[str, str] = {
    "freedom of speech":            "wolność słowa",
    "freedom of expression":        "wolność wypowiedzi",
    "freedom of assembly":          "wolność zgromadzeń",
    "freedom of religion":          "wolność religii",
    "human dignity":                "godność człowieka",
    "rule of law":                  "państwo prawa",
    "constitutional tribunal":      "Trybunał Konstytucyjny",
    "supreme court":                "Sąd Najwyższy",
    "constitution":                 "Konstytucja",
    "parliament":                   "parlament",
    "sejm":                         "Sejm",
    "senate":                       "Senat",
    "president":                    "Prezydent",
    "prime minister":               "Prezes Rady Ministrów",
    "council of ministers":         "Rada Ministrów",
    "ombudsman":                    "Rzecznik Praw Obywatelskich",
    "citizen":                      "obywatel",
    "citizenship":                  "obywatelstwo",
    "human rights":                 "prawa człowieka",
    "civil rights":                 "prawa obywatelskie",
    "personal liberty":             "wolność osobista",
    "right to privacy":             "prawo do prywatności",
    "right to property":            "prawo własności",
    "right to vote":                "prawo wyborcze",
    "state of emergency":           "stan wyjątkowy",
    "martial law":                  "stan wojenny",
    "referendum":                   "referendum",
    "amendment":                    "zmiana konstytucji",
    "judiciary":                    "władza sądownicza",
    "legislative power":            "władza ustawodawcza",
    "executive power":              "władza wykonawcza",
    "local government":             "samorząd terytorialny",
    "national security":            "bezpieczeństwo państwa",
    "public order":                 "porządek publiczny",
    "equality before the law":      "równość wobec prawa",
    "presumption of innocence":     "domniemanie niewinności",
    "fair trial":                   "sprawiedliwy proces",
    "article":                      "artykuł",
}
# fmt: on

# longest phrases first so "freedom of speech" wins over "freedom"
_PATTERN = re.compile(
    "|".join(re.escape(phrase) for phrase in sorted(LEGAL_GLOSSARY, key=len, reverse=True)),
    flags=re.IGNORECASE,
)


def apply_glossary(text: str) -> str:
    """Replace known English legal phrases with canonical Polish terms."""

    def _sub(match: re.Match[str]) -> str:
        return LEGAL_GLOSSARY[match.group(0).lower()]

    return _PATTERN.sub(_sub, text)
