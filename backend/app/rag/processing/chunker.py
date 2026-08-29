from collections.abc import Iterable
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Token-aware and genuinely recursive: it falls back through separators, so a single
# paragraph larger than chunk_size still gets split. A hand-rolled paragraph splitter
# emits that paragraph whole, and one 20k-token chunk blows past the embedding model's
# 8191-token input limit and fails the entire ingest.
_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    *,
    doc_id: str,
    source: str,
    tenant_id: str,
) -> list[dict[str, Any]]:
    """Chunk a document page by page.

    `pages` is (page_number, text). Chunking page-wise rather than concatenating the
    whole document first is what lets `page` reach the citation — concatenating loses
    the boundary, which is why citations used to render "Page 1" for every source.

    chunk_id is written into metadata as well as the top level, because metadata is
    what survives the round trip through Qdrant's payload; the point id itself is a
    uuid5 and carries no provenance a reader can use.
    """
    chunks: list[dict[str, Any]] = []
    for page_number, text in pages:
        if not text.strip():
            continue
        for chunk_index, body in enumerate(_splitter.split_text(text)):
            chunk_id = f"{doc_id}:p{page_number}:c{chunk_index}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": body,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "source": source,
                        "tenant_id": tenant_id,
                        "page": page_number,
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks
