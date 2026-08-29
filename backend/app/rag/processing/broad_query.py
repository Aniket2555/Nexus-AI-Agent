import re

# Rule-based, same posture as context_engine/intent_detector.py's detect_intent_rule_based
# (D4 — no LLM call for something this cheap to pattern-match). Deliberately narrow: a
# false positive here expands retrieval to a whole document unnecessarily, which costs
# more tokens/latency but isn't wrong the way a false-negative intent-routing match would
# be — so this errs toward including "all"/"every"/"entire" phrasing rather than trying to
# also catch every implicit way of asking for full coverage.
_BROAD_PATTERNS: list[re.Pattern] = [
    re.compile(r"\ball\b.{0,15}\b(content|courses?|sections?|topics?|chapters?|pages?)\b", re.I),
    re.compile(r"\bevery(thing)?\b.{0,15}\b(content|courses?|section|topic|chapter|page)?\b", re.I),
    re.compile(r"\b(entire|whole|complete|full)\b.{0,15}\b(pdf|document|file|content)\b", re.I),
    re.compile(r"\b(list|write|show|summarize)\s+(all|everything)\b", re.I),
]


def is_broad_document_query(text: str) -> bool:
    """True for a "give me full coverage of the document" style question.

    Real bug this exists to close: a per-chunk cross-encoder reranker scores "is this
    chunk relevant to the question," not "does this chunk belong to the document the
    user wants everything from." A document with several topically-distinct sections
    (e.g. one page per course in a multi-course curriculum PDF) can legitimately score
    some sections below the relevance threshold against a generic "write all the
    content" phrasing even though every section is exactly what's being asked for —
    measured live: two genuinely-relevant chunks from the same 3-page PDF scored 0.03
    and 0.11 against that exact phrasing, both below even an already-lowered 0.1
    threshold. Detecting this phrasing and bypassing per-chunk relevance filtering for
    the referenced document (retrieve/dense.py's `get_all_chunks`) is the fix; lowering
    the threshold further isn't, since that would also let genuinely irrelevant chunks
    through for every other kind of question.
    """
    return any(pattern.search(text) for pattern in _BROAD_PATTERNS)
