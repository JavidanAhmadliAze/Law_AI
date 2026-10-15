"""The candidate set is the UNION of each leg's top `per_leg`, not a fusion.

RRF decided the set with a score that penalised single-leg hits: a chunk one leg
found scored ~1/(k+r), a chunk both legs found mediocre scored ~2/(k+r), so a
provision at dense rank 1 could reach the reranker at fused rank 16 because BM25
never returned it. These pin the behaviour that replaced it.
"""

from law_ai.services.opensearch.client import OpenSearchService


def _hit(doc_id: str, score: float = 1.0, text: str = "t") -> dict:
    return {"_id": doc_id, "_score": score, "_source": {"text": text, "metadata": {}}}


def test_both_legs_contribute_their_own_top_n() -> None:
    dense = [_hit("d1"), _hit("d2"), _hit("d3")]
    sparse = [_hit("s1"), _hit("s2"), _hit("s3")]
    union, _ = OpenSearchService._union({"dense": dense, "bm25": sparse}, per_leg=2)
    # two from each leg, and the third of each left out
    assert {r.chunk.chunk_id for r in union} == {"d1", "d2", "s1", "s2"}


def test_a_chunk_in_both_legs_appears_once() -> None:
    """So the union is AT MOST 2 x per_leg, and smaller where the legs agree."""
    union, _ = OpenSearchService._union(
        {"dense": [_hit("a"), _hit("both")], "bm25": [_hit("both"), _hit("b")]}, per_leg=2
    )
    ids = [r.chunk.chunk_id for r in union]
    assert len(ids) == len(set(ids)) == 3


def test_a_single_leg_hit_is_not_penalised() -> None:
    """The whole point. Under RRF "solo" — rank 1 in one leg only — lost to
    "both", mediocre in each. Here rank 1 wins on its own rank."""
    union, _ = OpenSearchService._union(
        {"dense": [_hit("solo"), _hit("both")], "bm25": [_hit("x"), _hit("both")]}, per_leg=5
    )
    assert union[0].chunk.chunk_id in {"solo", "x"}
    assert [r.chunk.chunk_id for r in union].index("solo") < [
        r.chunk.chunk_id for r in union
    ].index("both")


def test_rank_order_within_a_single_leg_is_preserved() -> None:
    union, _ = OpenSearchService._union(
        {"dense": [_hit("first"), _hit("second"), _hit("third")], "bm25": []}, per_leg=5
    )
    assert [r.chunk.chunk_id for r in union] == ["first", "second", "third"]


def test_empty_legs() -> None:
    union, legs = OpenSearchService._union({"dense": [], "bm25": []}, per_leg=15)
    assert union == []
    assert legs == {}


def test_reports_which_leg_found_each_chunk() -> None:
    """The per-leg record is the diagnosis for a provision that went missing —
    whether both legs ranked it badly, or one never returned it at all."""
    _, legs = OpenSearchService._union(
        {"dense": [_hit("a", 0.9), _hit("both", 0.8)], "bm25": [_hit("both", 14.2)]}, per_leg=5
    )
    assert set(legs["both"]) == {"dense", "bm25"}
    assert set(legs["a"]) == {"dense"}
    assert legs["both"]["bm25"] == {"rank": 1, "score": 14.2}
    assert legs["both"]["dense"]["rank"] == 2


def test_tolerates_hits_without_a_score() -> None:
    union, legs = OpenSearchService._union(
        {"dense": [{"_id": "x", "_source": {"text": "t"}}]}, per_leg=5
    )
    assert [r.chunk.chunk_id for r in union] == ["x"]
    assert legs["x"]["dense"]["score"] == 0.0
