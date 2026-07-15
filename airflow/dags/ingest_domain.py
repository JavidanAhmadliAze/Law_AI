"""Offline ingestion DAG factory — one DAG per legal domain, thin tasks.

For every domain in the act registry (law_ai.acts) this module generates a
DAG `ingest_<domain>` whose tasks fan out over the domain's acts via dynamic
task mapping:

    fetching → pdf_parsing → chunking → metadata_creating → indexing_and_storing

Artifacts move between tasks as files under DATA_DIR named by act_id (XCom
carries act ids, never payloads). Indexing is idempotent (deterministic chunk
ids), so any DAG can be re-run safely.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

from law_ai.acts import acts_for_domain, domains

DATA_DIR = Path("/opt/airflow/data")


def _build_domain_dag(domain: str) -> None:
    @dag(
        dag_id=f"ingest_{domain}",
        start_date=datetime(2026, 1, 1),
        schedule=None,  # trigger manually (or set a cron for re-syncs)
        catchup=False,
        tags=["law-ai", "offline", domain],
        # ISAP intermittently answers with a bot challenge (fetcher raises
        # FetchError) — spaced retries ride it out
        default_args={"retries": 3, "retry_delay": timedelta(minutes=2)},
    )
    def ingest() -> None:
        @task
        def act_ids() -> list[str]:
            return [act.act_id for act in acts_for_domain(domain)]

        @task
        def fetching(act_id: str) -> str:
            from law_ai.acts import get_act
            from law_ai.config import Settings
            from law_ai.services.fetcher.factory import create_fetcher

            fetcher = create_fetcher(Settings())
            pdf_bytes = asyncio.run(fetcher.fetch(get_act(act_id).url))

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            (DATA_DIR / f"{act_id}.pdf").write_bytes(pdf_bytes)
            return act_id

        @task
        def pdf_parsing(act_id: str) -> str:
            from law_ai.acts import get_act
            from law_ai.config import Settings
            from law_ai.services.pdf_parser.factory import create_pdf_parser

            parser = create_pdf_parser(Settings())
            document = parser.parse(
                (DATA_DIR / f"{act_id}.pdf").read_bytes(), source_url=get_act(act_id).url
            )
            (DATA_DIR / f"{act_id}.txt").write_text(document.text)
            return act_id

        @task
        def chunking(act_id: str) -> str:
            from law_ai.config import Settings
            from law_ai.services.chunking.factory import create_chunker

            chunker = create_chunker(Settings())
            raw_chunks = chunker.chunk((DATA_DIR / f"{act_id}.txt").read_text())
            (DATA_DIR / f"{act_id}.raw_chunks.json").write_text(
                json.dumps([c.model_dump() for c in raw_chunks])
            )
            return act_id

        @task
        def metadata_creating(act_id: str) -> str:
            from law_ai.acts import get_act
            from law_ai.config import Settings
            from law_ai.services.chunking.base import RawChunk
            from law_ai.services.metadata.factory import create_metadata_builder

            builder = create_metadata_builder(Settings())
            raw_chunks = [
                RawChunk(**item)
                for item in json.loads((DATA_DIR / f"{act_id}.raw_chunks.json").read_text())
            ]
            law_chunks = builder.build(raw_chunks, act=get_act(act_id))
            (DATA_DIR / f"{act_id}.law_chunks.json").write_text(
                json.dumps([c.model_dump() for c in law_chunks])
            )
            return act_id

        @task
        def indexing_and_storing(act_id: str) -> int:
            """Embeds (inside the search service) + upserts into OpenSearch."""
            from law_ai.config import Settings
            from law_ai.schemas.chunk import LawChunk
            from law_ai.services.embedding.factory import create_embedder
            from law_ai.services.opensearch.factory import create_search_service

            settings = Settings()
            chunks = [
                LawChunk(**item)
                for item in json.loads((DATA_DIR / f"{act_id}.law_chunks.json").read_text())
            ]

            async def _run() -> int:
                embedder = create_embedder(settings)
                search = create_search_service(settings, embedder)
                await search.startup()
                try:
                    return await search.index_chunks(chunks)
                finally:
                    await search.teardown()

            return asyncio.run(_run())

        ids = act_ids()
        fetched = fetching.expand(act_id=ids)
        parsed = pdf_parsing.expand(act_id=fetched)
        chunked = chunking.expand(act_id=parsed)
        enriched = metadata_creating.expand(act_id=chunked)
        indexing_and_storing.expand(act_id=enriched)

    ingest()


for _domain in domains():
    _build_domain_dag(_domain)
