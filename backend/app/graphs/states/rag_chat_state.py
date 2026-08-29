from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    retrieved_docs: list[dict[str, Any]]
    compressed_context: str
    compression_stats: dict[str, Any]
    citations: list[dict[str, Any]]
    query_text: str

    recalled_memory: dict[str, Any]
    intent: dict[str, Any]
    rewritten_queries: list[str]
    system_prompt: str
    token_allocations: dict[str, int]

    draft_answer: AnyMessage | None
    quality_score: float
    reflection_feedback: str
    iteration_count: int
    reflection_cost_usd: float
