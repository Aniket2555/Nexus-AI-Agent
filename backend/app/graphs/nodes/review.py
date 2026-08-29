import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from backend.app.graphs.specialists.report import format_results
from backend.app.graphs.states.shared_state import SpecialistResult
from backend.app.llm.provider import get_chat_model
from backend.app.loop_engine.budget import estimate_cost_usd
from backend.app.utils.messages import as_text
from backend.app.utils.tokens import count_tokens

REVIEW_PROMPT = (
    "You are the quality reviewer in a multi-agent system (distinct from a "
    "general quality score — you are specifically checking grounding). Check "
    "the draft response below against the specialist results it is supposed to "
    "be based on.\n\n"
    "Flag it as FAIL if:\n"
    "- It states something not supported by any specialist result or citation.\n"
    "- It silently presents an incomplete picture as complete when a specialist "
    "failed.\n"
    "- A cited [Source: ...] does not correspond to an actual citation in the "
    "results.\n\n"
    "Respond in exactly this format, nothing else:\n"
    "VERDICT: PASS or FAIL\n"
    "FEEDBACK: <one sentence>\n\n"
    "Specialist results:\n{results}\n\n"
    "Draft response:\n{draft}\n"
)

_VERDICT = re.compile(r"VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)
_FEEDBACK = re.compile(r"FEEDBACK:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_review(text: str) -> tuple[bool, str]:
    """An unparseable review defaults to `False` (fail), not a fake pass — same
    "could not verify -> don't ship it" posture as Phase 3's
    `parse_quality_score()` defaulting to 0.0.
    """
    verdict_match = _VERDICT.search(text)
    passed = bool(verdict_match) and verdict_match.group(1).upper() == "PASS"

    feedback_match = _FEEDBACK.search(text)
    feedback = feedback_match.group(1).strip() if feedback_match else text.strip()

    return passed, feedback


async def review_draft(
    draft: str,
    agent_results: list[SpecialistResult],
    *,
    chat_model: BaseChatModel | None = None,
) -> tuple[bool, str, float]:
    """Reuses report.py's `format_results` (§6.1) rather than a second results
    formatter — the reviewer needs to see results in exactly the shape the report
    specialist synthesized the draft from, so a citation the draft makes can
    actually be checked against them.

    Returns a cost estimate too, same shape as Phase 3's `evaluate_response()`
    (`loop_engine/reflection.py`) — the supervisor's `cost_usd_spent` tracks review
    calls the same way `rag_chat`'s reflection loop tracks judge calls.
    """
    model = chat_model or get_chat_model(temperature=0)
    prompt = REVIEW_PROMPT.format(results=format_results(agent_results), draft=draft)

    response = await model.ainvoke([SystemMessage(content=prompt)])
    text = as_text(response)

    passed, feedback = parse_review(text)
    cost_usd = estimate_cost_usd(count_tokens(prompt), count_tokens(text))

    return passed, feedback, cost_usd
