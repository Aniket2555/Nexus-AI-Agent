import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from backend.app.llm.provider import get_chat_model
from backend.app.utils.messages import as_text

Intent = Literal["qa", "code", "analysis", "search", "creative"]
INTENTS: tuple[Intent, ...] = ("qa", "code", "analysis", "search", "creative")

# Ordered: first match wins. Deliberately narrow patterns (word boundaries, specific
# phrasing) over broad keyword hits — a false rule match routes a query to the wrong
# prompt template, which is worse than falling through to the LLM fallback.
_RULES: list[tuple[re.Pattern, Intent]] = [
    (
        re.compile(
            r"\b(def |class |function|traceback|stack trace|syntax error|"
            r"compile(r|s)?\b|refactor|write (a |the )?(function|script|class))",
            re.I,
        ),
        "code",
    ),
    (
        re.compile(
            r"(write (a|me)\b.{0,25}\b(poem|story|song|haiku)\b|"
            r"\b(brainstorm|come up with|imagine if|creative)\b)",
            re.I,
        ),
        "creative",
    ),
    (
        re.compile(
            r"\b(compare|trend|correlat(e|ion)|why did|root cause|analy[sz]e|"
            r"what caused|statistics? (on|about))\b",
            re.I,
        ),
        "analysis",
    ),
    (
        re.compile(r"\b(find|search for|look up|where can I find|locate)\b", re.I),
        "search",
    ),
    # Catch-all for direct factual questions. Checked last, after every more specific
    # category has had a chance — "where can I find X" hits `search` above before
    # this ever sees it, since a plain "where ...?" would otherwise match here too.
    (
        re.compile(
            r"^(who|what|when|where|why|how (many|much|do|does|is|are)|"
            r"is|are|does|do|can|could|would|should|will)\b",
            re.I,
        ),
        "qa",
    ),
]

INTENT_PROMPT = """Classify this question into exactly one category: {intents}.
Output only the category name, nothing else.

Question: {query}"""


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    method: Literal["rule", "llm", "fallback"]


def detect_intent_rule_based(text: str) -> Intent | None:
    for pattern, intent in _RULES:
        if pattern.search(text):
            return intent
    return None


class IntentDetector:
    """Rule-based classifier with an LLM fallback (§3.1) — not a routing decision by
    itself. Nothing branches to a different graph yet (that's Phase 6's specialist
    agents); this labels `state["intent"]` for prompt_assembler to pick a per-intent
    instruction block, and for future phases to route on.
    """

    def __init__(self, chat_model: BaseChatModel | None = None) -> None:
        # Lazy, not eager — see QueryRewriter's __init__ for why. It matters more
        # here than anywhere else in §3.1: most queries hit a rule and never touch
        # the LLM at all, so constructing an IntentDetector must never require a key.
        self._chat_model = chat_model

    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is None:
            self._chat_model = get_chat_model(temperature=0)
        return self._chat_model

    async def detect(self, text: str) -> IntentResult:
        rule_hit = detect_intent_rule_based(text)
        if rule_hit is not None:
            return IntentResult(intent=rule_hit, confidence=1.0, method="rule")

        response = await self.chat_model.ainvoke(
            INTENT_PROMPT.format(intents=", ".join(INTENTS), query=text)
        )
        candidate = as_text(response).strip().lower()
        for intent in INTENTS:
            if intent in candidate:
                return IntentResult(intent=intent, confidence=0.6, method="llm")

        # Most rag_chat traffic is QA-shaped; an unparseable LLM response defaults
        # there rather than raising, since intent is an optimization, not a gate.
        return IntentResult(intent="qa", confidence=0.0, method="fallback")
