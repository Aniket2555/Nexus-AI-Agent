from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CORPUS_PATH = Path("tests/eval/fixtures/corpus.yaml")
DEFAULT_GOLDEN_SET_PATH = Path("tests/eval/fixtures/golden_set.yaml")


@dataclass
class GoldenQuestion:
    id: str
    question: str
    expected_chunk_ids: list[str]


@dataclass
class Corpus:
    doc_id: str
    source: str
    tenant_id: str
    pages: list[tuple[int, str]]


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> Corpus:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Corpus(
        doc_id=data["doc_id"],
        source=data["source"],
        tenant_id=data["tenant_id"],
        pages=[(p["page"], p["text"]) for p in data["pages"]],
    )


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET_PATH) -> list[GoldenQuestion]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        GoldenQuestion(
            id=item["id"],
            question=item["question"],
            expected_chunk_ids=item["expected_chunk_ids"],
        )
        for item in data
    ]
