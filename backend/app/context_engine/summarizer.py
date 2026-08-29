from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage

from backend.app.config import get_settings
from backend.app.llm.provider import get_chat_model
from backend.app.utils.messages import as_text

SUMMARY_PROMPT = (
    "Summarize this conversation excerpt in 2-4 sentences, preserving any facts, "
    "decisions, or constraints a later turn might need to refer back to. Do not "
    "add commentary or a preamble — output only the summary itself.\n\n{transcript}"
)


@dataclass
class SummarizationResult:
    summary: str | None
    kept_messages: list[AnyMessage]

    @property
    def was_summarized(self) -> bool:
        return self.summary is not None


class ConversationSummarizer:
    """Rolling-window compaction (§3.1 stage 3): once a thread grows past
    `summarizer_trigger_message_count`, everything except the last
    `summarizer_keep_last_n` messages is collapsed into one summary — so `memory`'s
    share of the token budget (TokenBudget.memory_budget) stays roughly constant
    instead of growing with thread length.

    This does not touch LangGraph state persistence (DECISIONS.md D6 — the graph
    still owns no checkpointer); it only affects what goes into the *prompt* for the
    current turn.
    """

    def __init__(self, chat_model: BaseChatModel | None = None) -> None:
        self._chat_model = chat_model

    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is None:
            self._chat_model = get_chat_model(temperature=0)
        return self._chat_model

    async def maybe_summarize(self, messages: list[AnyMessage]) -> SummarizationResult:
        settings = get_settings()
        if len(messages) <= settings.summarizer_trigger_message_count:
            return SummarizationResult(summary=None, kept_messages=messages)

        keep_last_n = settings.summarizer_keep_last_n
        to_summarize = messages[:-keep_last_n]
        kept = messages[-keep_last_n:]
        if not to_summarize:
            return SummarizationResult(summary=None, kept_messages=kept)

        transcript = "\n".join(
            f"{m.type}: {as_text(m)}" for m in to_summarize if as_text(m)
        )
        response = await self.chat_model.ainvoke(SUMMARY_PROMPT.format(transcript=transcript))
        summary = as_text(response).strip() or None
        return SummarizationResult(summary=summary, kept_messages=kept)
