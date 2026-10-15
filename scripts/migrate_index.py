"""Rebuild the chunk index with the Polish analyzer and corrected article labels.

Two fixes land together because both need a new index and neither needs the
embedding model:

  1. analysis-stempel ("polish" analyzer). An analyzer is fixed for the life of
     an index, so switching it means creating a new one.
  2. the chunker's article regex. It truncated superscripts ("Art. 556[1]" filed
     as "Art. 556") and multi-letter suffixes ("Art. 117ba" as "Art. 117b"),
     collapsing 557 distinct articles onto their neighbours.

The chunker fix changed which characters are CAPTURED, never where an article
boundary falls, so chunk text is byte-identical to what is already indexed and
the stored vectors are still correct. A full re-ingest would re-embed 28k chunks
to arrive at the same numbers — hours of CPU for nothing.

So the vectors never leave the cluster. Server-side _reindex copies every
document into the new index, and only the chunks whose label actually changes
(a few hundred, not 28k) are then rewritten from the client. Pulling all
documents through Python is what an earlier version did, and 28k 1024-dim
vectors as JSON exhausted an 8 GB machine.

Labels are corrected IN PLACE, keeping each document's existing id. Minting a
new id (chunk_id hashes the article) would mean delete+insert, and chunks whose
text is identical — "(uchylony)", page furniture — cannot be told apart by text,
so several documents collapse onto one id and the rest are silently lost. An
earlier version did exactly that and dropped 993 chunks. Ids therefore no longer
match what a fresh ingest of the same act would mint; a fresh ingest rebuilds the
index anyway, and losing documents is the worse trade.

    uv run python -m scripts.migrate_index --target law-chunks-v2
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx

from law_ai.config import Settings
from law_ai.services.chunking.client import ArticleChunker

_SOURCE_DIR = Path("data")  # act .txt files, as written by the ingestion DAG


def _chunk_id(act: str, article: str, position: int, text: str) -> str:
    digest = hashlib.sha256(f"{act}|{article}|{position}|{text}".encode()).hexdigest()[:16]
    return f"{act}-{digest}"


async def _act_docs(client: httpx.AsyncClient, index: str, act: str) -> list[dict[str, Any]]:
    """Every document of one act. Scoped per act so the vectors of the whole
    corpus are never resident at once."""
    body = {"size": 500, "query": {"term": {"metadata.act": act}}}
    r = (await client.post(f"/{index}/_search?scroll=5m", json=body)).json()
    scroll_id, docs = r["_scroll_id"], list(r["hits"]["hits"])
    while True:
        r = (
            await client.post("/_search/scroll", json={"scroll": "5m", "scroll_id": scroll_id})
        ).json()
        hits = r["hits"]["hits"]
        if not hits:
            await client.request("DELETE", "/_search/scroll", json={"scroll_id": [scroll_id]})
            return docs
        docs.extend(hits)


def _corrected(source: Path) -> dict[str, tuple[str, str]]:
    """text -> (correct article, chapter), for texts UNAMBIGUOUS within the act.

    A text appearing under two different articles cannot be assigned by text
    alone, so it is dropped rather than guessed: those are "(uchylony)" stubs and
    page furniture, where a wrong label costs more than a stale one.

    Uses the chunker's own defaults, which is what create_chunker does, so the
    boundaries reproduce exactly what was indexed.
    """
    seen: dict[str, tuple[str, str]] = {}
    ambiguous: set[str] = set()
    for chunk in ArticleChunker().chunk(source.read_text()):
        value = (chunk.article, chunk.chapter)
        if chunk.text in seen and seen[chunk.text] != value:
            ambiguous.add(chunk.text)
        seen[chunk.text] = value
    return {text: value for text, value in seen.items() if text not in ambiguous}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", default=None)
    parser.add_argument("--target", default="law-chunks-v2")
    parser.add_argument("--source-dir", default=str(_SOURCE_DIR))
    args = parser.parse_args()

    settings = Settings()
    src_index = args.source_index or settings.opensearch.index
    base = f"http://{settings.opensearch.host}:{settings.opensearch.port}"
    src_dir = Path(args.source_dir)

    async with httpx.AsyncClient(base_url=base, timeout=600) as client:
        plugins = {p["component"] for p in (await client.get("/_cat/plugins?format=json")).json()}
        analyzer = "polish" if "analysis-stempel" in plugins else "standard"
        print(f"analyzer: {analyzer}")

        mapping = (await client.get(f"/{src_index}/_mapping")).json()[src_index]["mappings"]
        await client.delete(f"/{args.target}")
        created = await client.put(
            f"/{args.target}",
            json={
                "settings": {
                    "index": {"knn": True},
                    "analysis": {"analyzer": {"default": {"type": analyzer}}},
                },
                "mappings": mapping,
            },
        )
        created.raise_for_status()
        print(f"created {args.target}")

        # server-side copy: documents and their vectors never enter this process
        r = await client.post(
            "/_reindex?wait_for_completion=true&refresh=true",
            json={"source": {"index": src_index, "size": 500}, "dest": {"index": args.target}},
        )
        r.raise_for_status()
        copied = r.json()
        print(f"reindexed {copied['created']} docs, failures={len(copied.get('failures', []))}")

        acts = [
            b["key"]
            for b in (
                await client.post(
                    f"/{args.target}/_search",
                    json={
                        "size": 0,
                        "aggs": {"a": {"terms": {"field": "metadata.act", "size": 200}}},
                    },
                )
            ).json()["aggregations"]["a"]["buckets"]
        ]

        relabelled = rechaptered = no_source = ambiguous = 0
        for act in sorted(acts):
            source = src_dir / f"{act}.txt"
            if not source.exists():
                no_source += 1
                continue
            corrected = _corrected(source)
            docs = await _act_docs(client, args.target, act)
            actions: list[str] = []
            for doc in docs:
                body = doc["_source"]
                match = corrected.get(body["text"])
                if match is None:
                    ambiguous += 1
                    continue
                article, chapter = match
                meta = body["metadata"]
                patch: dict[str, str] = {}
                if article != meta["article"]:
                    patch["article"] = article
                    relabelled += 1
                if chapter and chapter != meta.get("chapter", ""):
                    patch["chapter"] = chapter
                    rechaptered += 1
                if not patch:
                    continue
                # only these fields move; the vector stays where it is
                actions.append(json.dumps({"update": {"_index": args.target, "_id": doc["_id"]}}))
                actions.append(json.dumps({"doc": {"metadata": patch}}))
            for i in range(0, len(actions), 1000):
                rr = await client.post(
                    "/_bulk",
                    content="\n".join(actions[i : i + 1000]) + "\n",
                    headers={"Content-Type": "application/x-ndjson"},
                )
                rr.raise_for_status()
                if rr.json().get("errors"):
                    bad = next(x for x in rr.json()["items"] if next(iter(x.values())).get("error"))
                    raise RuntimeError(f"bulk error: {bad}")
            del docs, actions

        await client.post(f"/{args.target}/_refresh")
        count = (await client.get(f"/{args.target}/_count")).json()["count"]
        print(
            f"\n{args.target}: {count} chunks, {relabelled} relabelled, "
            f"{rechaptered} chapters set, {ambiguous} skipped as ambiguous, "
            f"{no_source} acts without source text"
        )


if __name__ == "__main__":
    asyncio.run(main())
