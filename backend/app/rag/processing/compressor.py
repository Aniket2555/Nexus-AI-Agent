import re
from dataclasses import dataclass
from functools import lru_cache

from backend.app.config import get_settings
from backend.app.utils.tokens import count_tokens as _count_tokens

DEFAULT_COMPRESSOR_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


@dataclass
class CompressionResult:
    text: str
    original_tokens: int
    compressed_tokens: int
    method: str

    @property
    def ratio(self) -> float:
        return self.original_tokens / max(self.compressed_tokens, 1)


@lru_cache(maxsize=1)
def _load_llmlingua():
    """Best-effort load of LLMLingua-2.

    Returns None if it can't be loaded — no network for the first model download,
    or the package missing — so the compressor degrades to extractive summarization
    instead of failing the whole generate step. This is the fallback the plan (§2.3)
    calls for explicitly, not an ad hoc addition.
    """
    try:
        from llmlingua import PromptCompressor
    except ImportError:
        return None
    try:
        return PromptCompressor(DEFAULT_COMPRESSOR_MODEL, use_llmlingua2=True, device_map="cpu")
    except Exception:
        return None


def _extractive_compress(context: str, query: str, token_budget: int) -> str:
    """Fallback compression: keep the sentences with the most query-term overlap, in
    their original order, until the token budget is spent. No model, no download,
    always available — this is what runs if LLMLingua-2 can't be loaded.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    if not sentences:
        return context
    query_terms = {t.lower() for t in re.findall(r"\w+", query)}

    def overlap(sentence: str) -> int:
        return len(query_terms & {t.lower() for t in re.findall(r"\w+", sentence)})

    ranked = sorted(range(len(sentences)), key=lambda i: overlap(sentences[i]), reverse=True)
    keep = set()
    used_tokens = 0
    for i in ranked:
        cost = _count_tokens(sentences[i])
        if used_tokens + cost > token_budget and keep:
            continue
        keep.add(i)
        used_tokens += cost
        if used_tokens >= token_budget:
            break

    return " ".join(sentences[i] for i in sorted(keep))


def compress_context(context: str, query: str, token_budget: int | None = None) -> CompressionResult:
    """Compress an assembled context block to fit `token_budget`.

    A no-op when the context already fits — compression is lossy, so it is only
    applied when the token budget actually requires it.
    """
    settings = get_settings()
    budget = token_budget or settings.compression_token_budget
    original_tokens = _count_tokens(context)

    if original_tokens <= budget:
        return CompressionResult(context, original_tokens, original_tokens, "none")

    compressor = _load_llmlingua()
    if compressor is not None:
        result = compressor.compress_prompt([context], question=query, target_token=budget)
        compressed_text = result["compressed_prompt"]
        return CompressionResult(
            compressed_text, original_tokens, _count_tokens(compressed_text), "llmlingua2"
        )

    compressed_text = _extractive_compress(context, query, budget)
    return CompressionResult(
        compressed_text, original_tokens, _count_tokens(compressed_text), "extractive"
    )
