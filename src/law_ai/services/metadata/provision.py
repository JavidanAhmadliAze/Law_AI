"""The §-level label for a chunk — recovered from its text, not its metadata.

The index stores one ARTICLE label per paragraph, so all eleven paragraphs of
art. 70 read "Art. 70" and cannot be told apart in any listing. The paragraph
marker lives only at the front of the chunk text: "§ 2. Za zaległości..." or, in
the acts that number with ustępy, "Art. 41. 1. Stawka podatku...".

Two steps rather than one pattern: a single regex lets the article number
backtrack, so "Art. 291b. Jeżeli..." matches "Art. 29" plus paragraph "1b" and
invents a provision that does not exist.
"""

import re

_ART_PREFIX = re.compile(r"^Art\.\s*\d+[a-z]*(?:\[\d+\])?\.\s*")
_PARAGRAPH = re.compile(r"^§\s*(\d+[a-z]*)\.")
_USTEP = re.compile(r"^(\d+[a-z]*)\.")


def provision_label(article: str, text: str) -> str:
    """ "Art. 116 § 2" rather than "Art. 116"."""
    body = _ART_PREFIX.sub("", " ".join(text.split()), count=1)
    if m := _PARAGRAPH.match(body):
        return f"{article} § {m.group(1)}"
    if m := _USTEP.match(body):
        return f"{article} ust. {m.group(1)}"
    return article
