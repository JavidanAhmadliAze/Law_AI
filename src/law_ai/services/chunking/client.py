"""Article- and paragraph-aware chunker for Polish statutes.

Splitting hierarchy:
1. "Art. N." boundaries — the natural retrieval unit for statutes;
   the current "Rozdział" (chapter) header is carried along.
2. Within an article, numbered paragraphs — constitution-style ustępy
   ("1.", "2a." …) or code-style paragrafy ("§ 1.", "§ 2." …) — a chunk
   NEVER mixes two paragraphs.
3. Only a paragraph longer than `max_chars` is sub-split (with overlap),
   and each piece still comes from that single paragraph.
"""

import re

from law_ai.logging import get_logger
from law_ai.services.chunking.base import BaseChunker, RawChunk

logger = get_logger(__name__)

_ARTICLE_RE = re.compile(r"^Art\.\s*(\d+[a-z]?)\.?", re.MULTILINE)
_CHAPTER_RE = re.compile(r"^Rozdzia[łl]\s+([IVXLC]+[a-z]?)", re.MULTILINE)
# numbered paragraph at the start of a line: "2.", "3a." (ustęp) or "§ 2." (paragraf)
_PARAGRAPH_RE = re.compile(r"^\s*(?:§\s*)?\d+[a-z]?\.\s", re.MULTILINE)
# a line that is ONLY the article header, e.g. "Art. 54."
_HEADER_ONLY_RE = re.compile(r"^Art\.\s*\d+[a-z]?\.?\s*$")


class ArticleChunker(BaseChunker):
    def __init__(self, max_chars: int = 2000, overlap_chars: int = 200) -> None:
        self._max_chars = max_chars
        self._overlap = overlap_chars

    def chunk(self, text: str) -> list[RawChunk]:
        chunks: list[RawChunk] = []
        position = 0

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
                pieces = self._split_long(paragraph, article="", chapter="", start_position=position)
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
        if len(text) <= self._max_chars:
            return [RawChunk(text=text, article=article, chapter=chapter, position=start_position)]
        parts: list[RawChunk] = []
        step = self._max_chars - self._overlap
        n = 0
        for offset in range(0, len(text), step):
            piece = text[offset : offset + self._max_chars]
            if not piece.strip():
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
