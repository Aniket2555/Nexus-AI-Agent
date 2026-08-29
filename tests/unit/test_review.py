from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.app.graphs.nodes import review as review_module
from backend.app.graphs.states.shared_state import SpecialistResult

RESULTS = [
    SpecialistResult(
        task_id="t1",
        specialist="research",
        summary="NEXUS uses Groq for chat.",
        success=True,
        citations=[{"source": "handbook.pdf", "page": 3, "chunk_id": "x"}],
        details={},
    )
]


def test_parse_review_passes_on_pass_verdict():
    passed, feedback = review_module.parse_review("VERDICT: PASS\nFEEDBACK: Looks grounded.")
    assert passed is True
    assert feedback == "Looks grounded."


def test_parse_review_fails_on_fail_verdict():
    passed, feedback = review_module.parse_review("VERDICT: FAIL\nFEEDBACK: Unsupported claim.")
    assert passed is False
    assert feedback == "Unsupported claim."


def test_parse_review_defaults_to_fail_when_unparseable():
    """Same 'could not verify -> don't ship it' posture as parse_quality_score()
    defaulting to 0.0 (loop_engine/reflection.py) — an unparseable verdict must
    never default to PASS."""
    passed, feedback = review_module.parse_review("the model said something unexpected")
    assert passed is False
    assert feedback == "the model said something unexpected"


async def test_review_draft_returns_verdict_and_cost(monkeypatch):
    model = GenericFakeChatModel(
        messages=iter([AIMessage("VERDICT: PASS\nFEEDBACK: All claims are cited.")])
    )
    monkeypatch.setattr(review_module, "get_chat_model", lambda *args, **kwargs: model)

    passed, feedback, cost_usd = await review_module.review_draft("draft text", RESULTS)

    assert passed is True
    assert feedback == "All claims are cited."
    assert cost_usd > 0


async def test_review_draft_uses_explicit_chat_model_when_given():
    model = GenericFakeChatModel(messages=iter([AIMessage("VERDICT: FAIL\nFEEDBACK: Nope.")]))

    passed, feedback, cost_usd = await review_module.review_draft(
        "draft text", RESULTS, chat_model=model
    )

    assert passed is False
    assert feedback == "Nope."
