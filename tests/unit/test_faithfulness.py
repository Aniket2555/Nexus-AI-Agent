from langchain_core.messages import AIMessage

from backend.app.evaluation.faithfulness import answer_correctness, faithfulness, relevance


class _FakeJudge:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = None

    async def ainvoke(self, prompt):
        self.last_prompt = prompt
        return AIMessage(self.response_text)


async def test_faithfulness_returns_none_without_a_judge():
    """No GROQ_API_KEY -> no judge -> unmeasured, not a fabricated 0.0."""
    assert await faithfulness("answer", "context", judge=None) is None


async def test_faithfulness_parses_judge_score():
    assert await faithfulness("answer", "context", judge=_FakeJudge("0.75")) == 0.75


async def test_faithfulness_clamps_out_of_range_scores():
    assert await faithfulness("a", "c", judge=_FakeJudge("1.5")) == 1.0
    assert await faithfulness("a", "c", judge=_FakeJudge("-0.5")) == 0.0


async def test_faithfulness_returns_none_on_unparseable_response():
    assert await faithfulness("a", "c", judge=_FakeJudge("not a number")) is None


async def test_relevance_returns_none_without_a_judge():
    assert await relevance("question", "context", judge=None) is None


async def test_relevance_parses_judge_score_and_includes_the_question_in_the_prompt():
    judge = _FakeJudge("0.9")

    score = await relevance("what is PTO?", "PTO context", judge)

    assert score == 0.9
    assert "what is PTO?" in judge.last_prompt


async def test_answer_correctness_returns_none_without_a_judge():
    assert await answer_correctness("answer", "reference", judge=None) is None


async def test_answer_correctness_parses_judge_score():
    score = await answer_correctness("got it right", "the reference", judge=_FakeJudge("1.0"))

    assert score == 1.0
