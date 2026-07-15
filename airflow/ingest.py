"""Run the offline ingestion pipeline directly, without the Airflow scheduler.

Same services and stages as dags/ingest_domain.py — useful for local runs and
debugging. Lives OUTSIDE dags/ on purpose: the scheduler imports every module
in dags/, and this CLI pulls in app-only dependencies.

Usage:
    uv run python airflow/ingest.py --domain civil            # all civil acts
    uv run python airflow/ingest.py --act kodeks-cywilny      # a single act
    uv run python airflow/ingest.py --domain civil --dry-run  # everything except indexing
"""

import argparse
import asyncio
import sys

from law_ai.acts import LegalAct, acts_for_domain, get_act
from law_ai.dependencies import get_settings
from law_ai.logging import get_logger, setup_logging
from law_ai.services.chunking.factory import create_chunker
from law_ai.services.embedding.factory import create_embedder
from law_ai.services.fetcher.factory import create_fetcher
from law_ai.services.metadata.factory import create_metadata_builder
from law_ai.services.opensearch.factory import create_search_service
from law_ai.services.pdf_parser.factory import create_pdf_parser

logger = get_logger("ingest")


async def main(acts: list[LegalAct], dry_run: bool) -> int:
    settings = get_settings()
    setup_logging(settings)

    fetcher = create_fetcher(settings)
    parser = create_pdf_parser(settings)
    chunker = create_chunker(settings)
    metadata_builder = create_metadata_builder(settings)

    all_chunks = []
    for act in acts:
        pdf_bytes = await fetcher.fetch(act.url)
        document = parser.parse(pdf_bytes, source_url=act.url)
        raw_chunks = chunker.chunk(document.text)
        law_chunks = metadata_builder.build(raw_chunks, act=act)
        logger.info("ingest.prepared", act=act.act_id, chunks=len(law_chunks))
        all_chunks.extend(law_chunks)

    if dry_run:
        for chunk in all_chunks[:3]:
            logger.info(
                "ingest.sample",
                act=chunk.metadata.act,
                article=chunk.metadata.article,
                chapter=chunk.metadata.chapter,
                preview=chunk.text[:80].replace("\n", " "),
            )
        return 0

    embedder = create_embedder(settings)
    search = create_search_service(settings, embedder)
    await search.startup()
    try:
        count = await search.index_chunks(all_chunks)
        logger.info("ingest.indexed", count=count)
    finally:
        await search.teardown()
    return 0


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    group = arg_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--domain", help="ingest every act of a domain, e.g. civil")
    group.add_argument("--act", help="ingest a single act by slug, e.g. kodeks-cywilny")
    arg_parser.add_argument("--dry-run", action="store_true")
    args = arg_parser.parse_args()

    selected = acts_for_domain(args.domain) if args.domain else [get_act(args.act)]
    if not selected:
        arg_parser.error(f"no acts registered for domain {args.domain!r}")
    sys.exit(asyncio.run(main(selected, args.dry_run)))
