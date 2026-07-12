from law_ai.services.opensearch.client import OpenSearchService


def _hit(doc_id: str, text: str = "t") -> dict:
    return {"_id": doc_id, "_source": {"text": text, "metadata": {}}}


def test_rrf_rewards_presence_in_both_legs() -> None:
    dense = [_hit("a"), _hit("b"), _hit("c")]
    sparse = [_hit("b"), _hit("d")]
    fused = OpenSearchService._rrf_fuse([dense, sparse])
    ids = [r.chunk.chunk_id for r in fused]
    # "b" appears in both legs → must outrank single-leg results
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_preserves_rank_order_within_single_leg() -> None:
    dense = [_hit("first"), _hit("second"), _hit("third")]
    fused = OpenSearchService._rrf_fuse([dense, []])
    assert [r.chunk.chunk_id for r in fused] == ["first", "second", "third"]


def test_rrf_empty_legs() -> None:
    assert OpenSearchService._rrf_fuse([[], []]) == []


def test_filter_builder() -> None:
    clause = OpenSearchService._build_filters({"article": "Art. 54", "chapter": ""})
    # empty values are dropped
    assert clause == [{"term": {"metadata.article": "Art. 54"}}]
    assert OpenSearchService._build_filters(None) == []
