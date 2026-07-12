from law_ai.acts import get_act
from law_ai.services.chunking.base import RawChunk
from law_ai.services.metadata.client import ActMetadataBuilder

_ACT = get_act("konstytucja")


def _raw(text: str = "Art. 1. Treść.", article: str = "Art. 1") -> RawChunk:
    return RawChunk(text=text, article=article, chapter="Rozdział I", position=0)


def test_chunk_ids_are_deterministic() -> None:
    builder = ActMetadataBuilder()
    first = builder.build([_raw()], act=_ACT)
    second = builder.build([_raw()], act=_ACT)
    assert first[0].chunk_id == second[0].chunk_id  # idempotent re-indexing


def test_different_content_different_id() -> None:
    builder = ActMetadataBuilder()
    a = builder.build([_raw(text="Art. 1. A.")], act=_ACT)
    b = builder.build([_raw(text="Art. 1. B.")], act=_ACT)
    assert a[0].chunk_id != b[0].chunk_id


def test_different_act_different_id() -> None:
    builder = ActMetadataBuilder()
    a = builder.build([_raw()], act=_ACT)
    b = builder.build([_raw()], act=get_act("kodeks-cywilny"))
    assert a[0].chunk_id != b[0].chunk_id


def test_metadata_fields_populated() -> None:
    chunk = ActMetadataBuilder().build([_raw()], act=_ACT)[0]
    assert chunk.metadata.article == "Art. 1"
    assert chunk.metadata.chapter == "Rozdział I"
    assert chunk.metadata.language == "pl"
    assert chunk.metadata.act == "konstytucja"
    assert chunk.metadata.domain == "constitutional"
    assert chunk.metadata.source_url == _ACT.url
    assert chunk.metadata.effective_date == "1997-10-17"
