from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.app.loop_engine.reflection import evaluate_response, parse_quality_score, should_retry


def test_parse_score_from_strict_format():
    assert parse_quality_score("SCORE: 0.9\nFEEDBACK: looks good") == 0.9


def test_parse_score_case_insensitive_and_extra_whitespace():
    assert parse_quality_score("score:   0.42  \nfeedback: ok") == 0.42


def test_parse_score_falls_back_to_any_bare_decimal():
    """Models don't always follow formatting instructions exactly."""
    assert parse_quality_score("I'd rate this about 0.75 overall, pretty solid.") == 0.75


def test_parse_score_defaults_to_zero_not_a_fake_pass_when_unparseable():
    """Regression test for the original sketch's bug: parse_quality_score was called
    but never defined (guaranteed NameError). The fixed version must fail toward
    'needs a retry', not silently toward 'good enough'."""
    assert parse_quality_score("This response is pretty good, no notes.") == 0.0


def test_parse_score_clamps_out_of_range_values():
    assert parse_quality_score("SCORE: 1.5") == 1.0


async def test_evaluate_response_returns_parsed_score_and_feedback():
    judge = GenericFakeChatModel(
        messages=iter([AIMessage("SCORE: 0.3\nFEEDBACK: missing a citation")])
    )

    result = await evaluate_response("some answer", "some context", chat_model=judge)

    assert result.quality_score == 0.3
    assert "missing a citation" in result.feedback
    assert result.cost_usd > 0


def test_should_retry_below_threshold():
    assert should_retry(iteration_count=0, quality_score=0.5, cumulative_cost_usd=0.0) == "retry"


def test_should_retry_above_threshold_responds():
    assert should_retry(iteration_count=0, quality_score=0.95, cumulative_cost_usd=0.0) == "respond"


def test_should_retry_stops_at_max_iterations_even_if_quality_is_low():
    assert should_retry(iteration_count=3, quality_score=0.1, cumulative_cost_usd=0.0) == "respond"


def test_should_retry_stops_at_cost_ceiling_even_if_quality_is_low():
    result = should_retry(0, 0.1, 999.0)

    assert result == "respond"
