import logging
import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from backend.app.config import get_settings
from backend.app.llm.provider import get_chat_model
from backend.app.loop_engine.budget import estimate_cost_usd
from backend.app.utils.messages import as_text
from backend.app.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

REFLECTION_PROMPT = (
    "Evaluate the following response for quality.\n\n"
    "Criteria:\n"
    "1. Completeness — Does it fully answer the question, OR — if the context "
    "genuinely does\n"
    "   not contain the answer — does it say so plainly instead of guessing? Both "
    "count as\n"
    '   complete. An honest "the context doesn\'t cover this" is a SUCCESS, not a '
    "shortfall:\n"
    "   the failure mode this criterion actually guards against is inventing an "
    "answer the\n"
    "   context doesn't support, not admitting the context's limits.\n"
    "2. Accuracy — Are claims factually grounded in the provided context?\n"
    "3. Hallucination-free — Does it avoid making unsupported claims?\n"
    "4. Citation-grounded — Are sources properly cited when a real claim is being "
    "made? A\n"
    "   response that correctly declines to answer needs no citation at all — do "
    "not penalize\n"
    "   the absence of one there.\n\n"
    "If the response honestly declines to answer because the context is "
    "insufficient, score it\n"
    "0.8 or higher unless it also does one of the things above wrong (e.g. it "
    "still fabricates\n"
    "a citation, or hedges by guessing anyway). Regenerating the same non-answer "
    "against the\n"
    "same context will not produce a more complete one — reserve a low score for "
    "cases a retry\n"
    "could plausibly fix.\n\n"
    "Respond in exactly this format, nothing else:\n"
    "SCORE: <a number between 0.0 and 1.0>\n"
    "FEEDBACK: <one sentence on what, if anything, needs improvement>\n\n"
    "Response to evaluate:\n{response}\n\nContext used:\n{context}\n"
)

_SCORE_LINE = re.compile(r"SCORE:\s*([01](?:\.\d+)?)", re.I)
_ANY_DECIMAL_0_1 = re.compile(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b")


def parse_quality_score(text: str) -> float:
    """Extract the 0.0-1.0 quality score from the judge's response.

    The original sketch called `parse_quality_score(evaluation.content)` without
    ever defining the function — a guaranteed NameError the first time
    reflection_node actually ran, caught here by writing (and running) the
    function it was missing rather than assuming the sketch was complete.

    Two attempts: the strict `SCORE: <n>` line the prompt asks for, then a loose
    fallback for any bare 0-1 decimal in the text (models don't always follow
    formatting instructions exactly). Defaults to 0.0 — not a fake passing score —
    when neither matches: an unparseable judgement means quality could not be
    verified, which should route to a retry rather than silently pass.
    """
    match = _SCORE_LINE.search(text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))

    match = _ANY_DECIMAL_0_1.search(text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))

    logger.warning("Could not parse a quality score from reflection judge output: %r", text[:200])
    return 0.0


@dataclass
class ReflectionResult:
    quality_score: float
    feedback: str
    cost_usd: float


async def evaluate_response(
    response: str, context: str, *, chat_model: BaseChatModel | None = None
) -> ReflectionResult:
    """Self-evaluate a generated response against the context it was grounded in."""
    judge = chat_model or get_chat_model(temperature=0)
    prompt = REFLECTION_PROMPT.format(response=response, context=context)

    evaluation = await judge.ainvoke([SystemMessage(content=prompt)])
    feedback = as_text(evaluation)

    return ReflectionResult(
        quality_score=parse_quality_score(feedback),
        feedback=feedback,
        cost_usd=estimate_cost_usd(count_tokens(prompt), count_tokens(feedback)),
    )


def should_retry(
    iteration_count: int, quality_score: float, cumulative_cost_usd: float
) -> Literal["retry", "respond"]:
    """Conditional edge: retry generation if quality is below threshold, unless the
    iteration cap or the cost ceiling (§3.2 "Budget Tracking") has been hit — either
    stop condition wins over a low score, so a genuinely hard question can't loop
    forever or run up an unbounded bill chasing a threshold it may never clear.
    """
    settings = get_settings()

    if iteration_count >= settings.reflection_max_iterations:
        return "respond"
    if cumulative_cost_usd >= settings.max_reflection_cost_usd:
        return "respond"
    if quality_score < settings.reflection_quality_threshold:
        return "retry"
    return "respond"
