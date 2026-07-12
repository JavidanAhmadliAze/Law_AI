from law_ai.services.chunking.client import ArticleChunker

SAMPLE = """\
Rozdział I
RZECZPOSPOLITA

Art. 1. Rzeczpospolita Polska jest dobrem wspólnym wszystkich obywateli.

Art. 2. Rzeczpospolita Polska jest demokratycznym państwem prawnym.

Rozdział II
WOLNOŚCI, PRAWA I OBOWIĄZKI

Art. 30. Przyrodzona i niezbywalna godność człowieka stanowi źródło wolności.
"""

# article header on its own line + two numbered paragraphs (ustępy)
MULTI_PARAGRAPH = """\
Art. 54.
1. Każdemu zapewnia się wolność wyrażania swoich poglądów oraz pozyskiwania
i rozpowszechniania informacji.
2. Cenzura prewencyjna środków społecznego przekazu oraz koncesjonowanie
prasy są zakazane.
"""

# paragraph 1 embedded in the header line ("Art. 54. 1. ...")
INLINE_FIRST_PARAGRAPH = """\
Art. 54. 1. Każdemu zapewnia się wolność wyrażania swoich poglądów.
2. Cenzura prewencyjna środków społecznego przekazu jest zakazana.
"""


def test_splits_on_articles() -> None:
    chunks = ArticleChunker().chunk(SAMPLE)
    articles = [c.article for c in chunks]
    assert articles == ["Art. 1", "Art. 2", "Art. 30"]


def test_tracks_chapters() -> None:
    chunks = ArticleChunker().chunk(SAMPLE)
    by_article = {c.article: c.chapter for c in chunks}
    assert by_article["Art. 1"] == "Rozdział I"
    assert by_article["Art. 30"] == "Rozdział II"


def test_paragraphs_are_never_mixed() -> None:
    chunks = ArticleChunker().chunk(MULTI_PARAGRAPH)
    assert len(chunks) == 2
    assert all(c.article == "Art. 54" for c in chunks)
    # each chunk holds exactly one ustęp
    assert "wolność wyrażania" in chunks[0].text
    assert "Cenzura" not in chunks[0].text
    assert "Cenzura" in chunks[1].text
    assert "wolność wyrażania" not in chunks[1].text


def test_header_only_line_stays_with_first_paragraph() -> None:
    chunks = ArticleChunker().chunk(MULTI_PARAGRAPH)
    assert chunks[0].text.startswith("Art. 54.")


def test_inline_first_paragraph_is_own_chunk() -> None:
    chunks = ArticleChunker().chunk(INLINE_FIRST_PARAGRAPH)
    assert len(chunks) == 2
    # paragraph 1 (inside the header line) must not be merged into paragraph 2
    assert chunks[0].text.startswith("Art. 54. 1.")
    assert "Cenzura" not in chunks[0].text
    assert chunks[1].text.startswith("2.")


# code-style paragrafy ("§ 1.") as in Kodeks cywilny
CODE_PARAGRAPHS = """\
Art. 155. § 1. Umowa sprzedaży, zamiany, darowizny przenosi własność na nabywcę.
§ 2. Jeżeli przedmiotem umowy są rzeczy oznaczone tylko co do gatunku,
do przeniesienia własności potrzebne jest przeniesienie posiadania rzeczy.
"""


def test_code_style_paragraphs_are_split() -> None:
    chunks = ArticleChunker().chunk(CODE_PARAGRAPHS)
    assert len(chunks) == 2
    assert all(c.article == "Art. 155" for c in chunks)
    assert chunks[0].text.startswith("Art. 155. § 1.")
    assert chunks[1].text.startswith("§ 2.")
    assert "gatunku" not in chunks[0].text


def test_long_paragraph_is_subsplit_within_itself() -> None:
    long_text = "Art. 5. " + "x" * 5000
    chunks = ArticleChunker(max_chars=2000, overlap_chars=200).chunk(long_text)
    assert len(chunks) > 1
    assert all(c.article == "Art. 5" for c in chunks)
    assert all(len(c.text) <= 2000 for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_positions_are_sequential_across_articles() -> None:
    chunks = ArticleChunker().chunk(SAMPLE)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_no_articles_falls_back_to_plain_paragraphs() -> None:
    text = "first paragraph\n\nsecond paragraph"
    chunks = ArticleChunker().chunk(text)
    assert len(chunks) == 2
    assert chunks[0].text == "first paragraph"
    assert chunks[1].text == "second paragraph"
