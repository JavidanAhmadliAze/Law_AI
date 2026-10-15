"""Re-split indexed chunks that swallowed a footnoted paragraph, then reindex them.

    uv run python scripts/resplit_merged_paragraphs.py            # report only
    uv run python scripts/resplit_merged_paragraphs.py --apply    # rewrite the index

Before the splitter learned about footnote markers, "§ 2.4)" did not start a new
paragraph, so art. 299 § 2 KSH — the exoneration defence — sat inside the § 1
chunk and no query could return it under its own label. Those chunks are already
in the index; this re-splits the affected ARTICLES with the current chunker and
replaces them, instead of re-ingesting whole acts (~0.8 chunks/sec on CPU).

An article is rebuilt from the text of its own chunks, joined in provision order,
because that text is the parsed article body — the same input the chunker had.
"""

import argparse
import asyncio
import re
from collections import defaultdict
from typing import Any

from law_ai.acts import ACTS
from law_ai.dependencies import get_settings
from law_ai.services.chunking.factory import create_chunker
from law_ai.services.embedding.factory import create_embedder
from law_ai.services.metadata.client import ActMetadataBuilder
from law_ai.services.opensearch.factory import create_search_service

# a chunk still holding a second paragraph marker inside its text never split
INNER = re.compile(r"(?:(?<=\.\s)|(?<=\)\s))(?:§\s*)?\d+[a-z]?\.\d+\)")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write to the index")
    parser.add_argument("--act", default="", help="limit to one act slug")
    args = parser.parse_args()

    settings = get_settings()
    search = create_search_service(settings, create_embedder(settings))
    await search.startup()
    chunker = create_chunker(settings)   # the same chunker ingestion uses
    builder = ActMetadataBuilder()
    acts = {a.act_id: a for a in ACTS}
    try:
        client = search._require_client()
        response = await client.search(
            index=settings.opensearch.index,
            body={"size": 10000, "_source": ["text", "metadata"],
                  "query": ({"term": {"metadata.act": args.act}} if args.act
                            else {"match_all": {}})},
        )
        hits = response["hits"]["hits"]
        affected: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for hit in hits:
            text = " ".join((hit["_source"].get("text") or "").split())
            if INNER.search(text):
                meta = hit["_source"]["metadata"]
                affected[(meta["act"], meta["article"])].append(hit)
        print(f"scanned {len(hits)} chunks · articles to re-split: {len(affected)}")

        for (act_id, article), merged in sorted(affected.items()):
            same = [h for h in hits
                    if h["_source"]["metadata"]["act"] == act_id
                    and h["_source"]["metadata"]["article"] == article]
            # ordered by the paragraph number, read as a number: sorting the
            # provision strings puts "§ 10" between "§ 1" and "§ 2" and rebuilds
            # the article in the wrong order
            def para_key(hit: dict[str, Any]) -> tuple[int, str]:
                label = hit["_source"]["metadata"].get("provision", "")
                m = re.search(r"(?:§|ust\.)\s*(\d+)([a-z]*)", label)
                return (int(m.group(1)), m.group(2)) if m else (0, "")

            body = "\n".join(h["_source"]["text"] for h in sorted(same, key=para_key))
            raw = [r for r in chunker.chunk(body) if r.article == article]
            new = builder.build(raw, act=acts[act_id]) if raw else []
            print(f"\n{act_id} {article}: {len(same)} chunk(s) in the index "
                  f"({len(merged)} merged) -> {len(new)} after re-splitting")
            for chunk in new:
                print(f"    {chunk.metadata.provision:<22}{chunk.text[:70]}")
            if args.apply and new:
                removed = [h["_id"] for h in same if h["_id"] not in {c.chunk_id for c in new}]
                indexed = await search.index_chunks(new)
                for chunk_id in removed:
                    await client.delete(index=settings.opensearch.index, id=chunk_id,
                                        ignore=[404])
                print(f"    indexed {indexed} · deleted {len(removed)} stale")
        if not args.apply:
            print("\n(report only — pass --apply to rewrite the index)")
    finally:
        await search.teardown()


if __name__ == "__main__":
    asyncio.run(main())
