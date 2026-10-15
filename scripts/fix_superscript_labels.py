"""Restore the bracket form of superscript article labels, chunk by chunk.

    uv run python scripts/fix_superscript_labels.py                 # report only
    uv run python scripts/fix_superscript_labels.py --apply         # write
    uv run python scripts/fix_superscript_labels.py --act kodeks-pracy

PDF extraction dropped the superscript for some acts, so art. 300¹³² of the KSH
is indexed as "Art. 300132" and art. 11¹ of the labour code as "Art. 111". Gold
written in the index's own bracket form ("Art. 300[132]") then matches nothing.

The split cannot be guessed from the label: "Art. 111" is art. 11¹ in one place
in the labour code and the real art. 111 in another, and "Art. 86" of the KRS is
a genuine article that a greedy rule would rewrite to "Art. 8[6]". The source
text decides it, by DOCUMENT ORDER — a superscript article sits between art. N
and art. N+1 — and each chunk is matched to its own occurrence by text, so the
two "Art. 111" occurrences are told apart.

Nothing is deleted and nothing is re-embedded: only `article` and `provision`
are rewritten, on chunks whose occurrence is unambiguous. A chunk that cannot be
placed is reported and left exactly as it is.
"""

import argparse
import asyncio
import re
from pathlib import Path

from law_ai.dependencies import get_settings
from law_ai.services.chunking.client import clean
from law_ai.services.embedding.factory import create_embedder
from law_ai.services.metadata.provision import provision_label
from law_ai.services.opensearch.factory import create_search_service

DATA = Path("data")
HEADER = re.compile(r"^Art\.\s*(\d+[a-z]*(?:\[\d+\])?)\.", re.MULTILINE)
MATCH_CHARS = 70          # of a chunk's text, enough to place it uniquely
flat = lambda t: " ".join(t.split())  # noqa: E731


def occurrences(text: str) -> list[tuple[str, str]]:
    """(label, body) per article, in document order, with superscripts restored.

    The label is inferred from order: after "Art. 300", a header reading
    "Art. 3001" can only be art. 300¹, because art. 3001 does not belong there.
    """
    heads = list(HEADER.finditer(text))
    out: list[tuple[str, str]] = []
    base, last_sup = None, 0
    for i, m in enumerate(heads):
        raw = m.group(1)
        body = text[m.start(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        label = f"Art. {raw}"
        if "[" in raw:
            base = raw.split("[")[0]
            last_sup = int(re.search(r"\[(\d+)\]", raw).group(1))
        else:
            num, letter = re.match(r"(\d+)([a-z]*)", raw).groups()
            suffix = num[len(base):] if base and num.startswith(base) and len(num) > len(base) else ""
            # ascending rather than strictly +1: the series carries gaps (a
            # repealed article) and lettered members (art. 300⁷¹ᵃ, indexed as
            # "30071a"), and demanding +1 abandoned the rest of the chapter from
            # the first such member — art. 300⁷² onwards stayed flattened.
            advances = suffix and not suffix.startswith("0") and (
                int(suffix) > last_sup or (int(suffix) == last_sup and letter))
            if advances:
                label = f"Art. {base}[{suffix}]{letter}"
                last_sup = int(suffix)
            else:
                base, last_sup = num, 0
        out.append((label, flat(clean(body))))
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the corrected labels")
    parser.add_argument("--act", default="", help="limit to one act slug")
    args = parser.parse_args()

    settings = get_settings()
    search = create_search_service(settings, create_embedder(settings))
    await search.startup()
    client = search._require_client()
    index = settings.opensearch.index
    try:
        for path in sorted(DATA.glob("*.txt")):
            act = path.stem
            if args.act and act != args.act:
                continue
            occ = occurrences(path.read_text(encoding="utf-8", errors="replace"))
            if not any("[" in label for label, _ in occ):
                continue                      # nothing to restore in this act

            before = await search.count_chunks(act=act)
            response = await client.search(
                index=index, body={"size": 10000, "_source": ["text", "metadata"],
                                   "query": {"term": {"metadata.act": act}}})
            hits = response["hits"]["hits"]

            # the index's flattened label -> the inferred label(s) it could mean.
            # Usually one, so every chunk under that label is settled without
            # looking at text at all; where a label means two different articles
            # ("Art. 111" is art. 11[1] in one place and art. 111 in another),
            # the chunk's own text picks between those candidates only.
            candidates: dict[str, list[tuple[str, str]]] = {}
            for label, body in occ:
                flat_label = label.replace("[", "").replace("]", "")
                candidates.setdefault(flat_label, []).append((label, body))

            planned: list[tuple[str, str, str, str]] = []   # id, article, provision, was
            unresolved: list[str] = []
            for hit in hits:
                meta = hit["_source"]["metadata"] or {}
                current = meta.get("article", "")
                options = candidates.get(current, [])
                if not options:
                    continue                       # label is not one the source renames
                if len({label for label, _ in options}) == 1:
                    label = options[0][0]
                else:
                    needle = flat(hit["_source"].get("text") or "")[:MATCH_CHARS]
                    hosts = {label for label, body in options if len(needle) >= 20 and needle in body}
                    if len(hosts) != 1:
                        unresolved.append(f'{current} ({hit["_id"][-8:]})')
                        continue
                    label = hosts.pop()
                if label != current:
                    planned.append((hit["_id"], label,
                                    provision_label(label, hit["_source"]["text"]), current))
            print(f"\n{act}: {len(hits)} chunks · {len(planned)} to relabel · "
                  f"{len(unresolved)} ambiguous (left untouched)")
            if unresolved:
                print(f"    ambiguous: {sorted(set(unresolved))[:6]}")
            for _, label, provision, was in planned[:5]:
                print(f"    {was!r} -> {label!r}   provision {provision!r}")
            if args.apply and planned:
                for chunk_id, label, provision, _ in planned:
                    # a doc merge, not the metadata helper: that helper's script
                    # also drops `institution`, which the planner's table of
                    # contents is built from
                    await client.update(index=index, id=chunk_id,
                                        body={"doc": {"metadata": {"article": label,
                                                                   "provision": provision}}})
                await client.indices.refresh(index=index)
                after = await search.count_chunks(act=act)
                print(f"    relabelled {len(planned)} · chunks before {before} after {after}"
                      + ("  ✓" if before == after else "  !! COUNT CHANGED"))
        if not args.apply:
            print("\n(report only — pass --apply to write)")
    finally:
        await search.teardown()


if __name__ == "__main__":
    asyncio.run(main())
