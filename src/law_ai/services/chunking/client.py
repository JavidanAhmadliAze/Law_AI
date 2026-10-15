"""Article- and paragraph-aware chunker for Polish statutes.

Splitting hierarchy:
1. "Art. N." boundaries — the natural retrieval unit for statutes;
   the current "Rozdział" (chapter) header is carried along.
2. Within an article, numbered paragraphs — constitution-style ustępy
   ("1.", "2a." …) or code-style paragrafy ("§ 1.", "§ 2." …) — a chunk
   NEVER mixes two paragraphs.
3. Only a paragraph longer than `max_chars` is sub-split (with overlap),
   and each piece still comes from that single paragraph.

Page furniture is stripped BEFORE any of that. The consolidated-text PDFs carry
a footer on every page, and it lands wherever the page broke — usually mid
sentence, sometimes as a chunk of its own.
"""

import re

from law_ai.logging import get_logger
from law_ai.services.chunking.base import BaseChunker, RawChunk

logger = get_logger(__name__)

# The superscript in an amended article ("Art. 556¹") extracts as "Art. 556[1]",
# and it is part of the article's identity, not decoration: Art. 556 makes the
# seller liable for defects while Art. 556[4] withdraws that whole section from
# consumers. Capturing only the digits before the bracket filed 216 Civil Code
# articles under their base neighbour — "Art. 709" alone absorbed the nineteen
# articles of the leasing chapter — which left the writer unable to tell them
# apart under one header, made an `article` metadata filter unable to name them,
# and let evaluation count a hit on 709[18] as a hit on 709.
#
# `[a-z]*` rather than `[a-z]?` for the same reason: amended articles carry
# multi-letter suffixes, heavily so in tax law, and a single optional letter
# truncated 341 of them — "Art. 117ba" filed as "Art. 117b", and the whole
# anti-avoidance run Art. 119a-119zfn collapsing onto a handful of labels.
_ARTICLE_RE = re.compile(r"^Art\.\s*(\d+[a-z]*(?:\s*\[\d+\])?)\.?", re.MULTILINE)
# Roman OR Arabic: the codes number chapters "Rozdzial I", the statutes
# "Rozdzial 1", "Rozdzial 1a". Matching only Roman left 86% of the corpus
# with no chapter at all — 24,763 of 28,783 chunks — so the one metadata
# field that could separate two subjects WITHIN an act was empty. It also
# governs where an article body ends, so an unmatched chapter heading let
# the next chapter's title leak into the previous article's last chunk.
_CHAPTER_RE = re.compile(r"^Rozdzia[łl]\s+([IVXLC]+[a-z]?|\d+[a-z]?)\b", re.MULTILINE)
# numbered paragraph at the start of a line: "2.", "3a." (ustęp) or "§ 2." (paragraf),
# optionally followed by the footnote markers the consolidated text prints on an
# amended paragraph: "§ 2.4)" and "§ 1.4),5)". Requiring whitespace straight
# after the dot silently merged those paragraphs into the one before them —
# art. 299 § 2 KSH, the whole exoneration defence, lived inside the § 1 chunk
# and no search could ever return it under its own label.
_PARAGRAPH_RE = re.compile(r"^\s*(?:§\s*)?\d+[a-z]?\.(?:\s*\d+\)[,;]?)*\s", re.MULTILINE)
# a line that is ONLY the article header, e.g. "Art. 54."
_HEADER_ONLY_RE = re.compile(r"^Art\.\s*\d+[a-z]*(?:\s*\[\d+\])?\.?\s*$")

# ---------------------------------------------------------------- cleaning ---
# The Sejm footer, printed on every page of the consolidated text and extracted
# inline: "©Kancelaria Sejmu s. 110/388 2026-09-09". It appears in 4,958 of the
# 28,783 indexed chunks. Two distinct harms, and both are fixed here rather than
# downstream because both are caused by WHERE it lands:
#
#   * it interrupts a sentence at the page break, so the chunk the reader and
#     the cross-encoder see reads "...nie będzie odpowiedzialny za szkodę.
#     ©Kancelaria Sejmu s. 97/243 2026-06-25"
#   * where a number happens to precede it on its own line, _PARAGRAPH_RE reads
#     that number as a paragraph start, so the chunker split at the page break
#     and emitted the remainder — number plus footer — as a chunk with no law
#     in it. Those chunks are dropped by `has_substance` once the mark is gone.
#
# In the PDF the mark, the page reference and the date sit on three separate
# lines, so `\s+` has to span newlines; all three parts are optional because the
# mark also extracts bare. Verified against every extracted act: 388 of 388
# occurrences in the ordynacja podatkowa matched, with no fragment left behind.
_FOOTER_RE = re.compile(
    r"©\s*Kancelaria\s+Sejmu(?:\s+s\.\s*\d+\s*/\s*\d+)?(?:\s+\d{4}-\d{2}-\d{2})?",
    re.IGNORECASE,
)
# Footnotes listing the amendments to a referenced act. They are editorial
# apparatus, not normative text, and they interleave into articles at page
# breaks — one sat inside art. 70 § 7 of the ordynacja podatkowa, between the
# provision's own points. Anchored to the whole sentence so it cannot eat the
# Dz. U. citations that appear INSIDE provisions.
_AMENDMENT_FOOTNOTE_RE = re.compile(
    r"\d+\)\s*Zmiany\s+(?:tekstu\s+jednolitego\s+)?wymienionej\s+ustawy\s+"
    r"(?:zostały\s+)?ogłoszone\s+w\s+Dz\.\s*U\..*?(?:\.|$)",
    re.IGNORECASE | re.DOTALL,
)
# Whitespace the removals leave behind, including a space before punctuation.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:)])")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SPACES_RE = re.compile(r"[ \t]{2,}")
# A chunk with no letter in it carries no law — a stranded number, or what is
# left of a footer-only chunk. This is deliberately the ONLY numeric rule: a
# regex deleting number-only lines looked right and was not. Of 3,260 such lines
# across the corpus, 571 were annex table cells and wrapped cross-references —
# the akcyza tariff annex numbers its rows "1", "2", "3" against "ex 1507",
# "ex 1508" — so the rule would have deleted real content in every annex.
_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def clean(text: str) -> str:
    """Strip page furniture from extracted statute text.

    Applied before splitting, so the footer cannot create a paragraph boundary
    where the page happened to break.
    """
    text = _FOOTER_RE.sub(" ", text)
    text = _AMENDMENT_FOOTNOTE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACES_RE.sub(" ", text)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def has_substance(text: str) -> bool:
    """Whether a chunk holds anything a query could match."""
    return bool(_HAS_LETTER_RE.search(text))


class ArticleChunker(BaseChunker):
    def __init__(self, max_chars: int = 2000, overlap_chars: int = 200) -> None:
        self._max_chars = max_chars
        self._overlap = overlap_chars

    def chunk(self, text: str) -> list[RawChunk]:
        chunks: list[RawChunk] = []
        position = 0

        text = clean(text)

        article_matches = list(_ARTICLE_RE.finditer(text))
        chapter_matches = list(_CHAPTER_RE.finditer(text))

        def chapter_at(offset: int) -> str:
            current = ""
            for m in chapter_matches:
                if m.start() <= offset:
                    current = f"Rozdział {m.group(1)}"
                else:
                    break
            return current

        if not article_matches:
            logger.warning("chunking.no_articles_found")
            for paragraph in self._split_paragraphs(text):
                pieces = self._split_long(
                    paragraph, article="", chapter="", start_position=position
                )
                chunks.extend(pieces)
                position += len(pieces)
            return chunks

        # an article body ends at the next article OR the next chapter header,
        # so "Rozdział ..." lines never leak into the preceding article's chunks
        boundaries = sorted(m.start() for m in article_matches + chapter_matches)

        for match in article_matches:
            start = match.start()
            end = next((b for b in boundaries if b > start), len(text))
            body = text[start:end].strip()
            if not body:
                continue
            article = f"Art. {match.group(1)}"
            chapter = chapter_at(start)

            for paragraph in self._split_paragraphs(body):
                pieces = self._split_long(
                    paragraph, article=article, chapter=chapter, start_position=position
                )
                chunks.extend(pieces)
                position += len(pieces)

        logger.info("chunking.done", articles=len(article_matches), chunks=len(chunks))
        return chunks

    # ------------------------------------------------------------ internals

    def _split_paragraphs(self, body: str) -> list[str]:
        """Split an article body into paragraphs (ustępy) — never merged."""
        starts = [m.start() for m in _PARAGRAPH_RE.finditer(body)]
        if not starts:
            # no numbered paragraphs — fall back to blank-line paragraphs
            parts = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            return parts or [body.strip()]

        paragraphs: list[str] = []
        prefix = body[: starts[0]].strip()
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else len(body)
            paragraphs.append(body[s:e].strip())

        if prefix:
            if _HEADER_ONLY_RE.match(prefix):
                # bare "Art. N." header — keep it with the first paragraph for context
                paragraphs[0] = f"{prefix}\n{paragraphs[0]}"
            else:
                # header line already contains paragraph 1 ("Art. 54. 1. …") —
                # it IS a paragraph of its own; never merge it into the next one
                paragraphs.insert(0, prefix)
        return paragraphs

    def _split_long(
        self, text: str, *, article: str, chapter: str, start_position: int
    ) -> list[RawChunk]:
        """Sub-split an over-long paragraph; pieces stay within the paragraph."""
        if not has_substance(text):
            return []
        if len(text) <= self._max_chars:
            return [RawChunk(text=text, article=article, chapter=chapter, position=start_position)]
        parts: list[RawChunk] = []
        step = self._max_chars - self._overlap
        n = 0
        for offset in range(0, len(text), step):
            piece = text[offset : offset + self._max_chars]
            if not has_substance(piece):
                continue
            parts.append(
                RawChunk(
                    text=piece,
                    article=article,
                    chapter=chapter,
                    position=start_position + n,
                )
            )
            n += 1
        return parts
