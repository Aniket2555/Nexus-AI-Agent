from langchain_core.messages import AIMessage

from backend.app.evaluation.cost_tracker import CostTracker, track_chat_model


def test_record_tokens_accumulates_totals():
    tracker = CostTracker()

    tracker.record_tokens(1000, 500)
    tracker.record_tokens(2000, 1000)

    assert tracker.total_input_tokens == 3000
    assert tracker.total_output_tokens == 1500
    assert tracker.call_count == 2
    assert tracker.total_cost_usd > 0


def test_record_tokens_returns_the_per_call_cost():
    tracker = CostTracker()

    cost = tracker.record_tokens(1000, 1000)

    assert cost == tracker.total_cost_usd


def test_cost_by_label_breaks_down_by_named_call_site():
    tracker = CostTracker()

    tracker.record_tokens(1000, 0, label="planner")
    tracker.record_tokens(1000, 0, label="planner")
    tracker.record_tokens(1000, 0, label="reviewer")

    assert tracker.cost_by_label["planner"] == 2 * tracker.cost_by_label["reviewer"]


def test_record_text_counts_tokens_itself():
    tracker = CostTracker()

    tracker.record_text("a prompt with several words", "a response")

    assert tracker.total_input_tokens > 0
    assert tracker.total_output_tokens > 0


def test_summary_rounds_and_serializes_cleanly():
    tracker = CostTracker()
    tracker.record_tokens(1000, 1000, label="x")

    summary = tracker.summary()

    assert summary["call_count"] == 1
    assert isinstance(summary["total_cost_usd"], float)
    assert "x" in summary["cost_by_label"]


class _FakeModel:
    async def ainvoke(self, prompt):
        return AIMessage("a response")


async def test_track_chat_model_meters_calls_transparently():
    tracker = CostTracker()
    wrapped = track_chat_model(_FakeModel(), tracker, label="judge")

    response = await wrapped.ainvoke("a prompt")

    assert response.content == "a response"
    assert tracker.call_count == 1
    assert tracker.cost_by_label["judge"] == tracker.total_cost_usd


async def test_track_chat_model_meters_every_call():
    tracker = CostTracker()
    wrapped = track_chat_model(_FakeModel(), tracker, label="judge")

    await wrapped.ainvoke("first")
    await wrapped.ainvoke("second")

    assert tracker.call_count == 2
