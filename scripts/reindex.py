"""Rebuild a versioned collection/index from source documents (§2.7).

    python -m scripts.reindex --source-dir data/documents --tenant-id acme --version v2
    python -m scripts.reindex --from-registry --tenant-id acme --version v2

Bumping `qdrant_collection` / `elasticsearch_index` to a new version and re-chunking
the original files is the supported way to change chunking strategy or embedding
model — mutating a live collection in place would leave it holding chunks from two
incompatible strategies, and the eval harness's numbers would stop meaning anything
(§2.7).

Two ways to supply the documents to re-chunk:

  --source-dir DIR   Original behavior: re-parse every supported file in DIR. Always
                      available, and the only option for files NEXUS never ingested
                      itself (a fresh corpus, someone else's export, etc).

  --from-registry     New: pull every successfully-ingested document for
                      --tenant-id out of the `document_registry` Postgres table
                      (`backend/app/models/database.py`) instead — no source
                      directory required. This is what closes the gap the previous
                      version of this docstring described: Phase 1/2 didn't persist
                      original uploaded bytes anywhere, so a reindex needed the
                      operator to have kept a matching `--source-dir` by hand. Every
                      upload through `POST /documents/upload` now also lands a row
                      (with content) in that table, so this flag can reconstruct
                      "every document ever ingested for this tenant" on its own.
                      Only documents whose registry row still has content (i.e.
                      ingested after the registry existed) are included — see
                      `list_documents_with_content()`'s docstring.
"""

import argparse
import asyncio
from pathlib import Path

from backend.app.models.database import list_documents_with_content
from backend.app.rag.ingestion.dispatch import SUPPORTED_SUFFIXES, extract_pages
from backend.app.rag.processing.chunker import chunk_pages
from backend.app.rag.retrieval.dense import DenseRetriever
from backend.app.rag.retrieval.sparse import SparseRetriever


async def _documents_from_source_dir(source_dir: Path) -> list[tuple[str, bytes]]:
    files = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        print(f"No supported files in {source_dir} (looked for {sorted(SUPPORTED_SUFFIXES)}).")
    return [(path.name, path.read_bytes()) for path in files]


async def _documents_from_registry(tenant_id: str) -> list[tuple[str, bytes]]:
    rows = await list_documents_with_content(tenant_id)
    if not rows:
        print(f"No registry documents with stored content for tenant {tenant_id!r}.")
    return [(row["filename"], row["content"]) for row in rows]


async def reindex(
    tenant_id: str,
    version: str,
    *,
    source_dir: Path | None = None,
    from_registry: bool = False,
) -> None:
    collection = f"nexus_documents_{version}"
    dense = DenseRetriever(collection_name=collection)
    sparse = SparseRetriever(index_name=collection)

    try:
        await dense.ensure_collection()
        await sparse.ensure_index()

        if from_registry:
            documents = await _documents_from_registry(tenant_id)
        else:
            assert source_dir is not None
            documents = await _documents_from_source_dir(source_dir)

        if not documents:
            return

        total_chunks = 0
        for filename, payload in documents:
            doc_id = f"{tenant_id}:{filename}"
            chunks = chunk_pages(
                extract_pages(filename, payload), doc_id=doc_id, source=filename, tenant_id=tenant_id
            )

            await dense.delete_document(doc_id)
            await sparse.delete_document(doc_id)

            await dense.upsert_chunks(chunks)
            await sparse.upsert_chunks(chunks)

            total_chunks += len(chunks)
            print(f"  {filename}: {len(chunks)} chunks")

        print(f"Reindexed {len(documents)} document(s), {total_chunks} chunks, into {collection!r}.")
        print("Update QDRANT_COLLECTION / ELASTICSEARCH_INDEX in .env to switch traffic to it.")
    finally:
        await dense.close()
        await sparse.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-dir", type=Path, help="Directory of source files to re-chunk."
    )
    source_group.add_argument(
        "--from-registry",
        action="store_true",
        help="Pull documents from the Postgres document_registry instead of a directory.",
    )

    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--version", required=True, help='e.g. "v2" -> nexus_documents_v2')

    args = parser.parse_args()

    asyncio.run(
        reindex(
            args.tenant_id,
            args.version,
            source_dir=args.source_dir,
            from_registry=args.from_registry,
        )
    )


if __name__ == "__main__":
    main()
