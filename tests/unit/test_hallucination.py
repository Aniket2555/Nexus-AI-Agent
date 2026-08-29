from langchain_core.messages import AIMessage

from backend.app.evaluation.hallucination import extract_claims, groundedness, hallucination_rate


class _ScriptedJudge:
    """Returns each response in `responses` in order — one call per LLM round
    trip, matching how `hallucination_rate` calls the judge twice (extract, then
    verdict).
    """

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(self._responses.pop(0))


async def test_extract_claims_parses_numbered_lines():
    judge = _ScriptedJudge("1. NEXUS uses LangGraph.\n2. PTO is 20 days.")

    claims = await extract_claims("some answer", judge)

    assert claims == ["NEXUS uses LangGraph.", "PTO is 20 days."]


async def test_extract_claims_returns_empty_for_a_claim_free_answer():
    judge = _ScriptedJudge("")

    assert await extract_claims("hello!", judge) == []


async def test_hallucination_rate_returns_none_without_a_judge():
    assert await hallucination_rate("answer", "context", judge=None) is None


async def test_hallucination_rate_returns_none_when_answer_has_no_claims():
    judge = _ScriptedJudge("")

    assert await hallucination_rate("hi there", "context", judge) is None


async def test_hallucination_rate_computes_fraction_unsupported():
    judge = _ScriptedJudge(
        "1. Claim A.\n2. Claim B.\n3. Claim C.",
        "1: SUPPORTED\n2: UNSUPPORTED\n3: SUPPORTED",
    )

    rate = await hallucination_rate("answer", "context", judge)

    assert rate == 1 / 3


async def test_hallucination_rate_treats_a_missing_verdict_as_unsupported():
    judge = _ScriptedJudge("1. Claim A.\n2. Claim B.", "1: SUPPORTED")

    rate = await hallucination_rate("answer", "context", judge)

    assert rate == 0.5


async def test_groundedness_is_the_complement_of_hallucination_rate():
    judge = _ScriptedJudge("1. Claim A.\n2. Claim B.", "1: SUPPORTED\n2: SUPPORTED")

    score = await groundedness("answer", "context", judge)

    assert score == 1.0


async def test_groundedness_returns_none_without_a_judge():
    assert await groundedness("answer", "context", judge=None) is None
