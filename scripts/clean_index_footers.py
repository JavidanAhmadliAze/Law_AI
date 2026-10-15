"""Strip the Sejm page footers from the chunks already in OpenSearch.

Only chunks that actually carry junk are re-embedded. The cleaner also collapses
whitespace the PDF extraction left behind, and chunks differing by nothing else
are recorded and skipped: re-embedding them would spend TEI time to move a
vector by almost nothing.

The chunker cleans this at ingestion now, but the live index was built before
that, so 17% of its chunks still carry "©Kancelaria Sejmu s. 110/388 2026-09-09"
inside the provision text — usually mid-sentence, where the page broke.

Four steps, and the first two run without touching the index:

    scan      every chunk through the ingestion cleaner, classified
    record    a manifest of exactly which chunks are affected, and how
    clean     the cleaned text re-embedded and upserted by chunk_id
    verify    a second scan, proving nothing is left

WHAT THIS CANNOT FIX. Cleaning a chunk is not the same as cleaning the document
before it was split. The footer sat at a page break, and where a number preceded
it the chunker read that as a paragraph start and split there — so a provision
interrupted by a page break is still two chunks, and re-ingesting the PDFs is
the only way to rejoin them. What this does fix is the text of every chunk and
the existence of the chunks that were nothing but furniture.

    uv run python scripts/clean_index_footers.py              # scan only
    uv run python scripts/clean_index_footers.py --apply      # then reindex
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from law_ai.config import Settings
from law_ai.schemas.chunk import ChunkMetadata, LawChunk
from law_ai.services.chunking.client import (
    _AMENDMENT_FOOTNOTE_RE,
    _FOOTER_RE,
    clean,
    has_substance,
)
from law_ai.services.embedding.factory import create_embedder
from law_ai.services.opensearch.factory import create_search_service


async def _scan(client: Any, index: str) -> tuple[list[dict[str, Any]], int]:
    """Every chunk in the index, classified by what cleaning does to it."""
    affected: list[dict[str, Any]] = []
    total = 0
    resp = await client.search(
        index=index,
        body={"query": {"match_all": {}}, "_source": ["chunk_id", "text", "metadata"]},
        size=500,
        scroll="5m",
    )
    sid, hits = resp["_scroll_id"], resp["hits"]["hits"]
    while hits:
        for h in hits:
            total += 1
            src = h["_source"]
            before = src["text"]
            after = clean(before)
            if after == before:
                continue
            md = src.get("metadata") or {}
            # Only JUNK earns a re-embed. `clean` also collapses double spaces
            # the PDF extraction left behind, and 588 chunks differ by nothing
            # else — re-embedding those spends TEI time to move a vector by
            # almost nothing. They are recorded and skipped.
            junk = bool(_FOOTER_RE.search(before) or _AMENDMENT_FOOTNOTE_RE.search(before))
            if not junk:
                action = "skip_whitespace_only"
            elif has_substance(after):
                action = "clean"
            else:
                action = "delete"
            affected.append(
                {
                    "chunk_id": src["chunk_id"],
                    "act": md.get("act", ""),
                    "article": md.get("article", ""),
                    "action": action,
                    "removed_chars": len(before) - len(after),
                    "before": " ".join(before.split())[:300],
                    "after": " ".join(after.split())[:300],
                    "metadata": md,
                }
            )
        resp = await client.scroll(scroll_id=sid, scroll="5m")
        hits = resp["hits"]["hits"]
    await client.clear_scroll(scroll_id=sid)
    return affected, total


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="reindex; without it, scan only")
    p.add_argument("--batch", type=int, default=200, help="chunks per reindex call")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/reports/footer_cleanup_manifest.json"),
    )
    a = p.parse_args()

    s = Settings()
    search = create_search_service(s, create_embedder(s))
    await search.startup()
    client = search._client  # noqa: SLF001 — maintenance script, no public scan API
    index = s.opensearch.index
    try:
        print(f"scanning {index} ...", flush=True)
        affected, total = await _scan(client, index)
        to_clean = [r for r in affected if r["action"] == "clean"]
        to_delete = [r for r in affected if r["action"] == "delete"]
        skipped = [r for r in affected if r["action"] == "skip_whitespace_only"]
        removed = sum(r["removed_chars"] for r in to_clean + to_delete)

        # The record, written before anything is modified: it is the only way to
        # tell afterwards which chunks were touched and what came off them.
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        a.manifest.write_text(
            json.dumps(
                {
                    "index": index,
                    "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "total_chunks": total,
                    "affected": len(affected),
                    "to_clean": len(to_clean),
                    "to_delete": len(to_delete),
                    "skipped_whitespace_only": len(skipped),
                    "junk_chars": removed,
                    "chunks": affected,
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        print(f"  {total:,} chunks scanned")
        print(f"  {len(affected):,} differ under the cleaner ({len(affected) / total:.1%})")
        print(f"    {len(to_clean):,} carry junk -> clean and re-embed")
        print(f"    {len(to_delete):,} are junk only -> delete")
        print(f"    {len(skipped):,} differ by whitespace only -> LEFT ALONE, not re-embedded")
        print(f"  {removed:,} junk characters")
        print(f"  manifest -> {a.manifest}")

        if not a.apply:
            print("\nscan only — rerun with --apply to reindex")
            return

        print(f"\nreindexing {len(to_clean):,} chunks (re-embedding as it goes)")
        started = time.monotonic()
        done = 0
        for i in range(0, len(to_clean), a.batch):
            rows = to_clean[i : i + a.batch]
            chunks = [
                LawChunk(
                    chunk_id=r["chunk_id"],
                    # cleaned from the CURRENT stored text, not from the truncated
                    # preview in the manifest
                    text="",
                    metadata=ChunkMetadata(**r["metadata"]),
                )
                for r in rows
            ]
            # fetch the full text again: the manifest only keeps a preview
            got = await client.mget(
                index=index, body={"ids": [r["chunk_id"] for r in rows]}, _source=["text"]
            )
            for chunk, doc in zip(chunks, got["docs"], strict=True):
                chunk.text = clean(doc["_source"]["text"])
            await search.index_chunks(chunks)
            done += len(chunks)
            left = (time.monotonic() - started) / done * (len(to_clean) - done)
            print(
                f"  {done:,}/{len(to_clean):,}  eta {int(left // 60)}m{int(left % 60):02d}s",
                flush=True,
            )

        if to_delete:
            print(f"\ndeleting {len(to_delete):,} chunks with no law left in them")
            for r in to_delete:
                await client.delete(index=index, id=r["chunk_id"], ignore=[404])
            await client.indices.refresh(index=index)

        print("\nverifying ...")
        still, total_after = await _scan(client, index)
        junk_left = [r for r in still if r["action"] != "skip_whitespace_only"]
        print(f"  {total_after:,} chunks now, {len(junk_left)} still carrying junk")
        print(
            f"  ({len(still) - len(junk_left):,} differ by whitespace only — left alone by design)"
        )
        if junk_left:
            print("  NOT CLEAN — remaining:", [r["chunk_id"] for r in junk_left[:5]])
    finally:
        await search.teardown()


if __name__ == "__main__":
    asyncio.run(main())
