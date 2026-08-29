from backend.app.config import get_settings
from backend.app.observability.metrics import llm_cost_usd_total, llm_tokens_total


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Illustrative per-call cost estimate for the reflection loop's budget-tracking
    conditional edge (§3.2). Uses the configured price-per-1k-token settings
    (`cost_per_1k_input_tokens_usd` / `cost_per_1k_output_tokens_usd`) rather than a
    real Groq billing API — Groq doesn't expose per-request cost in the response, so
    this is a token-count-based approximation, not a metered figure. Update the
    settings against console.groq.com/docs/pricing if the configured CHAT_MODEL
    changes (DECISIONS.md D7).

    This is the one function every LLM-cost call site in the project already
    goes through (review.py, reflection.py, evaluation/cost_tracker.py) — §8.6's
    "LLM Usage" dashboard metrics are recorded right here rather than at each
    call site individually, so nothing can call this and forget to report it.
    """
    settings = get_settings()
    cost = (input_tokens / 1000) * settings.cost_per_1k_input_tokens_usd + (
        output_tokens / 1000
    ) * settings.cost_per_1k_output_tokens_usd

    llm_cost_usd_total.inc(cost)
    llm_tokens_total.labels(direction="input").inc(input_tokens)
    llm_tokens_total.labels(direction="output").inc(output_tokens)

    return cost
