from langchain_core.language_models.chat_models import BaseChatModel

from backend.app.llm.provider import get_chat_model
from backend.app.utils.messages import as_text

HYDE_PROMPT = (
    "Write a short, plausible passage (2-4 sentences) that would directly answer "
    "this question, as if it were an excerpt from a real document. Do not hedge or "
    "say you don't know — invent a confident-sounding passage even if you're unsure. "
    "This is used only to improve semantic search, never shown to a user.\n\n"
    "Question: {query}\n\nPassage:"
)

STEP_BACK_PROMPT = (
    "Rewrite this specific question as a more general question about the broader "
    "topic or principle behind it — the kind of question whose answer would provide "
    "useful background for answering the original. Output only the rewritten "
    "question, nothing else.\n\nSpecific question: {query}\n\nGeneral question:"
)

MULTI_QUERY_PROMPT = (
    "Generate {n} different ways of phrasing this question, each on its own line "
    "with no numbering or bullets. Vary vocabulary and structure so together they "
    "cover more of the ways this question's answer might be phrased in a document. "
    "Output only the {n} lines.\n\nQuestion: {query}"
)


class QueryRewriter:
    """Three retrieval-query rewriting strategies (§3.1), each independently callable.

    None of these replace the original query — they're inputs the caller chooses to
    retrieve with *in addition to* it. `engine.py` wires multi-query in by default
    (broadest recall improvement for the least prompt-engineering risk); HyDE and
    step-back are available for callers that want them without needing a second
    implementation.
    """

    def __init__(self, chat_model: BaseChatModel | None = None) -> None:
        self._chat_model = chat_model

    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is None:
            self._chat_model = get_chat_model(temperature=0.3)
        return self._chat_model

    async def hyde(self, query: str) -> str:
        """Hypothetical Document Embeddings: retrieve using a fabricated answer's
        text instead of the question's — answer-shaped text often matches
        answer-shaped document passages better than question-shaped text does.
        """
        response = await self.chat_model.ainvoke(HYDE_PROMPT.format(query=query))
        return as_text(response) or query

    async def step_back(self, query: str) -> str:
        """A broader/more general version of the question, for retrieving background
        context a narrowly-matched search might miss."""
        response = await self.chat_model.ainvoke(STEP_BACK_PROMPT.format(query=query))
        return as_text(response) or query

    async def multi_query(self, query: str, n: int = 3) -> list[str]:
        """`n` paraphrases of the query. Falls back to just the original query
        (repeated) if the model's output can't be split into `n` non-empty lines —
        retrieval still works, it just doesn't get the recall benefit of variety.
        """
        response = await self.chat_model.ainvoke(MULTI_QUERY_PROMPT.format(query=query, n=n))
        lines = (line.strip("-* ") for line in as_text(response).splitlines())
        variants = [line for line in lines if line][:n]
        return variants or [query]
