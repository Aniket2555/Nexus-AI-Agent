from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AnyMessage

from backend.app.config import get_settings
from backend.app.context_engine.intent_detector import IntentDetector, IntentResult
from backend.app.context_engine.prompt_assembler import assemble_system_prompt
from backend.app.context_engine.query_rewriter import QueryRewriter
from backend.app.context_engine.summarizer import ConversationSummarizer
from backend.app.context_engine.token_budget import TokenBudget
from backend.app.utils.tokens import count_tokens


@dataclass
class AssembledPrompt:
    system_prompt: str
    intent: IntentResult
    memory_summary: str | None
    token_allocations: dict[str, int]


class ContextEngine:
    def __init__(
        self,
        rewriter: QueryRewriter | None = None,
        intent_detector: IntentDetector | None = None,
        summarizer: ConversationSummarizer | None = None,
        token_budget: TokenBudget | None = None,
    ) -> None:
        self.rewriter = rewriter or QueryRewriter()
        self.intent_detector = intent_detector or IntentDetector()
        self.summarizer = summarizer or ConversationSummarizer()
        self.token_budget = token_budget or TokenBudget(get_settings().context_window_tokens)

    async def expand_query(self, query: str) -> tuple[IntentResult, list[str]]:
        """Run before retrieval: classify intent and generate rewrite variants to
        search alongside the original query (§3.1 stages 1-2)."""
        intent = await self.intent_detector.detect(query)
        variants = await self.rewriter.multi_query(query, n=get_settings().multi_query_variants)
        return intent, variants

    async def assemble(
        self, query: str, context_block: str, messages: list[AnyMessage], intent: IntentResult
    ) -> AssembledPrompt:
        """Run after retrieval/rerank/compress: summarize older turns, allocate the
        token budget, and render the final system prompt (§3.1 stages 3, 5, 9).
        """
        summarization = await self.summarizer.maybe_summarize(messages)
        system_prompt = assemble_system_prompt(
            context_block, intent=intent.intent, memory_summary=summarization.summary
        )
        allocations = self.token_budget.allocate(
            system_tokens=count_tokens(system_prompt),
            memory_tokens=count_tokens(summarization.summary or ""),
            rag_tokens=count_tokens(context_block),
            query_tokens=count_tokens(query),
        )
        return AssembledPrompt(
            system_prompt=system_prompt,
            intent=intent,
            memory_summary=summarization.summary,
            token_allocations=allocations,
        )


def merge_query_variant_results(
    result_sets: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Union hybrid-search results across query rewrite variants, keeping the
    highest `rrf_score` seen for each chunk (a chunk two variants both surfaced is
    a stronger signal than either search alone, so it should keep its best rank).
    """
    best: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for doc in results:
            key = doc["metadata"].get("chunk_id", doc["id"])
            if key not in best or doc.get("rrf_score", 0) > best[key].get("rrf_score", 0):
                best[key] = doc
    return sorted(best.values(), key=lambda d: d.get("rrf_score", 0), reverse=True)
