"""Seed a fresh dev environment with a small, real sample corpus.

    python -m scripts.seed_data

`scripts/setup.sh` gets Qdrant/Postgres/Elasticsearch running and the Python
environment installed, but a brand-new dev stack has nothing in it to query —
running the RAG graph or hitting `POST /documents/upload` for the first time is
the only way to see anything come back, and that requires a document on hand.
This script ingests three short, made-up documents about a fictional open-source
project ("Project Lumen", a terrarium monitoring toolkit) through the exact same
code path `POST /documents/upload` uses — `extract_pages` -> `chunk_pages` ->
`DenseRetriever.upsert_chunks` / `SparseRetriever.upsert_chunks` — rather than
reimplementing any of that, so what lands in Qdrant/Elasticsearch here behaves
identically to a real upload. It also writes a `document_registry` row per
document via `backend/app/models/database.py`, the same as `documents.py`'s
`_ingest` now does, so `GET /documents` and `scripts/reindex.py --from-registry`
have something to show immediately.

This is a *different* corpus from the Phase 2.7 eval harness's fixed golden-set
corpus (a fictional 20-fact company handbook) — that one is a frozen baseline
other code measures against; reseeding dev data over it would quietly invalidate
every retrieval-quality number §2.7 records. Nothing here touches that fixture.
"""

import asyncio
import mimetypes

from backend.app.models.database import ensure_schema, record_document
from backend.app.rag.ingestion.dispatch import extract_pages
from backend.app.rag.processing.chunker import chunk_pages
from backend.app.rag.retrieval.dense import get_retriever
from backend.app.rag.retrieval.sparse import get_sparse_retriever

SEED_TENANT_ID = "default"

SAMPLE_DOCUMENTS: list[tuple[str, str]] = [
    (
        "lumen-overview.md",
        """# Project Lumen

Project Lumen is a fictional open-source toolkit for monitoring small indoor
terrariums. It pairs a low-power microcontroller with soil-moisture, temperature,
and light sensors, and reports readings to a self-hosted dashboard over Wi-Fi.

Lumen was started as a weekend project by a terrarium hobbyist who wanted to stop
guessing whether a bioactive vivarium's substrate had dried out. It grew into a
small community project with a plugin system for additional sensors (CO2,
humidity, soil pH) and a rules engine that can trigger a misting pump or grow
light automatically when a reading crosses a threshold.

The project has no commercial backing and no cloud service — every Lumen
installation runs entirely on the owner's own network, which the maintainers
consider a feature rather than a limitation.
""",
    ),
    (
        "lumen-hardware.md",
        """# Lumen Hardware Guide

A minimal Lumen node needs three things: a supported microcontroller, a sensor
set, and a 5V power supply. The reference build uses an ESP32-C3 board, chosen
for its built-in Wi-Fi and low idle power draw compared to older ESP32 variants.

## Supported sensors

- Capacitive soil moisture probes (avoid the cheaper resistive kind — they
  corrode within a few months in a humid enclosure).
- A combined temperature/humidity sensor, mounted away from direct light to
  avoid skewed temperature readings.
- A lux sensor for ambient light logging, used mostly to correlate plant growth
  with light exposure over time rather than for automation.

## Power

Nodes are designed to run continuously on USB power rather than battery, since
most terrarium setups already sit near an outlet for a grow light. A battery
mode exists for portable readings but is not the primary supported
configuration, and reduces reporting frequency to conserve power.
""",
    ),
    (
        "lumen-faq.md",
        """# Lumen FAQ

**Does Lumen require an internet connection?**
No. A Lumen node only needs to reach the local dashboard host over the LAN. The
dashboard itself never phones home, and the project has no cloud component at
all.

**Can one dashboard track multiple terrariums?**
Yes — each node reports with its own identifier, and the dashboard groups
readings per terrarium. There is no fixed limit on the number of nodes, though
the reference hardware has only been tested with up to about a dozen on one
Wi-Fi network.

**What happens if a sensor fails?**
The node keeps reporting other sensors and marks the failed reading as missing
rather than substituting a stale or zero value, so a dead soil-moisture probe
doesn't silently disable misting automation tied to temperature instead.

**Is there a mobile app?**
Not yet. The dashboard is a responsive web page, which the maintainers consider
sufficient for now given the project's size.
""",
    ),
]


async def seed() -> None:
    dense = get_retriever()
    sparse = get_sparse_retriever()

    await ensure_schema()
    await dense.ensure_collection()
    await sparse.ensure_index()

    total_chunks = 0
    try:
        for filename, text in SAMPLE_DOCUMENTS:
            payload = text.encode("utf-8")
            doc_id = f"{SEED_TENANT_ID}:{filename}"

            chunks = chunk_pages(
                extract_pages(filename, payload),
                doc_id=doc_id,
                source=filename,
                tenant_id=SEED_TENANT_ID,
            )

            await dense.delete_document(doc_id)
            await sparse.delete_document(doc_id)

            vectors = await dense.upsert_chunks(chunks)
            keywords = await sparse.upsert_chunks(chunks)

            await record_document(
                doc_id,
                SEED_TENANT_ID,
                filename,
                ".md",
                content_type=mimetypes.guess_type(filename)[0],
                status="succeeded",
                chunk_count=len(chunks),
                vector_count=vectors,
                keyword_count=keywords,
                content=payload,
            )

            total_chunks += len(chunks)
            print(f"  {filename}: {len(chunks)} chunks, {vectors} vectors, {keywords} keyword docs")
    finally:
        await dense.close()
        await sparse.close()

    print(f"Seeded {len(SAMPLE_DOCUMENTS)} document(s), {total_chunks} chunks, for tenant {SEED_TENANT_ID!r}.")
    print("Try: GET /api/v1/documents?tenant_id=default")


def main() -> None:
    asyncio.run(seed())


if __name__ == "__main__":
    main()
