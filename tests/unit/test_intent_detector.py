from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.app.context_engine.intent_detector import (
    IntentDetector,
    detect_intent_rule_based,
)

LABELLED_QUERIES = [
    ("How many days of PTO do I get?", "qa"),
    ("What is the capital of France?", "qa"),
    ("Who is responsible for the security review process?", "qa"),
    ("Write a Python function that reverses a linked list", "code"),
    ("I'm getting a stack trace when I run this script, can you help?", "code"),
    ("Refactor this class to use dependency injection", "code"),
    ("Can you write me a haiku about autumn?", "creative"),
    ("Brainstorm five names for a new product", "creative"),
    ("Write a short story about a robot learning to paint", "creative"),
    ("Compare Q1 and Q2 revenue and explain the trend", "analysis"),
    ("Why did the deployment fail last Tuesday?", "analysis"),
    ("Analyze the correlation between churn and support ticket volume", "analysis"),
    ("Find the onboarding document for new hires", "search"),
    ("Search for the latest vendor security review", "search"),
    ("Where can I find the Q3 budget spreadsheet?", "search"),
]


def test_rule_based_classification_meets_90_percent_accuracy_bar():
    correct = sum(
        1
        for query, expected in LABELLED_QUERIES
        if detect_intent_rule_based(query) == expected
    )
    accuracy = correct / len(LABELLED_QUERIES)
    assert accuracy >= 0.9, f"rule-based accuracy {accuracy:.0%} below the 90% bar"


async def test_rule_hit_never_calls_the_model():
    """A model that raises if invoked proves the rule path short-circuits."""

    class _ExplodingModel:
        async def ainvoke(self, *args, **kwargs):
            raise AssertionError("LLM should not be called when a rule matches")

    detector = IntentDetector(chat_model=_ExplodingModel())
    result = await detector.detect("Write a Python function that reverses a list")

    assert result.intent == "code"
    assert result.method == "rule"


async def test_llm_fallback_used_when_no_rule_matches():
    model = GenericFakeChatModel(messages=iter([AIMessage("qa")]))
    detector = IntentDetector(chat_model=model)

    result = await detector.detect("I'd like more details about the new pricing model.")

    assert result.intent == "qa"
    assert result.method == "llm"


async def test_unparseable_llm_response_falls_back_to_qa():
    model = GenericFakeChatModel(messages=iter([AIMessage("I'm not sure how to categorize this")]))
    detector = IntentDetector(chat_model=model)

    result = await detector.detect("some ambiguous input")

    assert result.intent == "qa"
    assert result.method == "fallback"
    assert result.confidence == 0.0
