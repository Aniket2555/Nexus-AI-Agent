from langchain_core.language_models.chat_models import BaseChatModel

from backend.app.utils.messages import as_text


def parse_score(text: str) -> float | None:
    """Parse a judge's single-line `"<number>"` response into a clamped [0, 1]
    float, or `None` if it didn't parse as a number at all.
    """
    try:
        score = float(text.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None

    return max(0.0, min(1.0, score))


async def judge_score(prompt: str, judge: BaseChatModel | None = None) -> float | None:
    """Shared plumbing for every single-number LLM-as-judge metric across this
    package (`faithfulness.py`, `agent_evaluator.py`) — one place, not three
    near-identical private copies.

    Returns `None` — never `0.0` — when there's no judge to ask, or when the
    judge's output doesn't parse as a number. A silent `0.0` would read as
    "measured and completely failing" in a report; `None` means "unmeasured,"
    which is what actually happened.
    """
    if judge is None:
        return None

    response = await judge.ainvoke(prompt)
    return parse_score(as_text(response))
