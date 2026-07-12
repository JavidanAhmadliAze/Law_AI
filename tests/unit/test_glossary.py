from law_ai.services.translation.glossary import apply_glossary


def test_replaces_known_terms() -> None:
    assert apply_glossary("freedom of speech") == "wolność słowa"


def test_longest_phrase_wins() -> None:
    # "freedom of speech" must not be partially replaced via a shorter entry
    result = apply_glossary("Is freedom of speech protected?")
    assert "wolność słowa" in result
    assert "freedom" not in result.lower()


def test_case_insensitive() -> None:
    assert "Trybunał Konstytucyjny" in apply_glossary("the CONSTITUTIONAL TRIBUNAL ruled")


def test_multiple_terms_in_one_sentence() -> None:
    result = apply_glossary("Does the constitution guarantee human dignity and rule of law?")
    assert "Konstytucja" in result
    assert "godność człowieka" in result
    assert "państwo prawa" in result


def test_text_without_terms_unchanged() -> None:
    text = "completely unrelated sentence"
    assert apply_glossary(text) == text
