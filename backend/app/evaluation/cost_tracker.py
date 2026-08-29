from dataclasses import dataclass, field
from typing import Any

from backend.app.loop_engine.budget import estimate_cost_usd
from backend.app.utils.messages import as_text
from backend.app.utils.tokens import count_tokens


@dataclass
class CostTracker:
    """Accumulates token/cost across an evaluation or benchmark run (§8.5).

    Reuses `loop_engine.budget.estimate_cost_usd`/`utils.tokens.count_tokens`
    rather than a second pricing implementation (DECISIONS.md D4) — this is pure
    bookkeeping on top of the same per-call estimate Phase 3's reflection loop and
    Phase 6's reviewer already use, not a new cost model.
    """

    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0
    cost_by_label: dict[str, float] = field(default_factory=dict)

    def record_tokens(self, input_tokens: int, output_tokens: int, *, label: str = "") -> float:
        cost = estimate_cost_usd(input_tokens, output_tokens)

        self.total_cost_usd += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

        if label:
            self.cost_by_label[label] = self.cost_by_label.get(label, 0.0) + cost

        return cost

    def record_text(self, prompt: str, response: str, *, label: str = "") -> float:
        """Convenience wrapper: counts tokens itself, for the common case of
        already having the raw prompt/response strings rather than pre-counted
        token counts.
        """
        return self.record_tokens(count_tokens(prompt), count_tokens(response), label=label)

    def summary(self) -> dict[str, float | int | dict[str, float]]:
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "call_count": self.call_count,
            "cost_by_label": {k: round(v, 6) for k, v in self.cost_by_label.items()},
        }


@dataclass
class _TrackedChatModel:
    """Wraps any object with an async `ainvoke(prompt) -> message`
    (`BaseChatModel` and every fake/judge double in this test suite both qualify)
    so every call is transparently metered into `tracker`, without threading a
    `CostTracker` parameter through `faithfulness.py`/`hallucination.py`'s metric
    functions — they only ever call `judge.ainvoke(...)`, so wrapping the judge
    once, at the call site that has a tracker to give it, is the whole
    integration.
    """

    model: Any
    tracker: CostTracker
    label: str = ""

    async def ainvoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        response = await self.model.ainvoke(prompt, *args, **kwargs)

        prompt_text = prompt if isinstance(prompt, str) else str(prompt)
        self.tracker.record_text(prompt_text, as_text(response), label=self.label)

        return response


def track_chat_model(model: Any, tracker: CostTracker, *, label: str = "") -> Any:
    return _TrackedChatModel(model, tracker, label=label)
