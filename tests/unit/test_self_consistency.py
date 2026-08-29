from backend.app.loop_engine import self_consistency as sc_module
from backend.app.loop_engine.self_consistency import (
    SelfConsistencyState,
    _normalize,
    self_consistency_graph,
    vote_node,
)


class _SharedFakeModel:
    """Fake model returning one response per call, in order. Constructed once and
    monkeypatched in as a shared instance — a `lambda: _SharedFakeModel(...)` that
    constructs fresh per call would reset the iterator on every one of the N
    parallel fan-out branches and defeat the whole test (verified the hard way while
    writing this: see the identical bug fixed in test_rag_chat_graph.py).
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(next(self._responses))


def test_normalize_collapses_case_whitespace_and_trailing_punctuation():
    assert _normalize("Paris.") == _normalize("PARIS") == _normalize("  paris  ")


async def test_vote_node_picks_the_majority_answer():
    state: SelfConsistencyState = {
        "question": "q",
        "context": "c",
        "samples": ["Paris", "Paris.", "London"],
        "final_answer": "",
        "vote_counts": {},
    }

    result = await vote_node(state)

    assert result["final_answer"] in ("Paris", "Paris.")
    assert result["vote_counts"]["paris"] == 2
    assert result["vote_counts"]["london"] == 1


async def test_vote_node_handles_no_samples():
    state: SelfConsistencyState = {
        "question": "q",
        "context": "c",
        "samples": [],
        "final_answer": "",
        "vote_counts": {},
    }

    result = await vote_node(state)

    assert result == {"final_answer": "", "vote_counts": {}}


async def test_fan_out_runs_n_parallel_samples_and_votes_the_majority(monkeypatch):
    """End-to-end through the real compiled graph: real Send-based fan-out, real
    operator.add accumulation across branches, real vote."""
    model = _SharedFakeModel(["Paris", "Paris.", "London"])
    monkeypatch.setattr(sc_module, "get_chat_model", lambda temperature=0.7: model)

    result = await self_consistency_graph.ainvoke(
        {"question": "capital of france?", "context": "France's capital is Paris.", "samples": []}
    )

    assert sorted(result["samples"]) == ["London", "Paris", "Paris."]
    assert result["vote_counts"] == {"paris": 2, "london": 1}
    assert result["final_answer"] in ("Paris", "Paris.")


async def test_all_distinct_samples_still_resolves_deterministically(monkeypatch):
    model = _SharedFakeModel(["Answer A", "Answer B", "Answer C"])
    monkeypatch.setattr(sc_module, "get_chat_model", lambda temperature=0.7: model)

    result = await self_consistency_graph.ainvoke({"question": "q", "context": "c", "samples": []})

    assert len(result["samples"]) == 3
    assert result["final_answer"] in result["samples"]
