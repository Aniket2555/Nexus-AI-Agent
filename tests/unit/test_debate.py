from langchain_core.messages import AIMessage

from backend.app.loop_engine import debate as debate_module
from backend.app.loop_engine.debate import debate_graph


class _RoutingFakeModel:
    """One shared instance for the whole graph run, routing by which unique prompt
    phrase appears — advocate/critic/moderator each get a distinct system prompt
    (per debate.py's *_PROMPT templates), so this can tell them apart reliably.
    """

    async def ainvoke(self, messages):
        text = messages[0].content if messages else ""

        if "arguing FOR the most likely answer" in text:
            return AIMessage("The context clearly states X, so the answer is X.")
        if "Critique the argument" in text:
            return AIMessage("The argument overstates certainty; the context only implies X.")
        if "Weigh the argument and the critique" in text:
            return AIMessage("Likely X, with moderate confidence given the critique.")

        raise AssertionError(f"unexpected prompt: {text[:80]}")


async def test_debate_runs_advocate_then_critic_then_moderator_in_order(monkeypatch):
    model = _RoutingFakeModel()
    monkeypatch.setattr(debate_module, "get_chat_model", lambda temperature=0.1: model)

    result = await debate_graph.ainvoke(
        {
            "question": "what is X?",
            "context": "The context implies X.",
            "advocate_argument": "",
            "critic_argument": "",
            "final_answer": "",
        }
    )

    assert "answer is X" in result["advocate_argument"]
    assert "overstates certainty" in result["critic_argument"]
    assert "moderate confidence" in result["final_answer"]


async def test_critic_receives_the_advocates_actual_argument(monkeypatch):
    """The critic's prompt must be built from advocate_argument, not a hardcoded
    stand-in — verified by making the fake model echo back what it was given."""

    class _EchoingModel:
        def __init__(self):
            self.critic_saw = None

        async def ainvoke(self, messages):
            text = messages[0].content if messages else ""

            if "arguing FOR the most likely answer" in text:
                return AIMessage("Distinctive advocate claim #42.")
            if "Critique the argument" in text:
                self.critic_saw = text
                return AIMessage("critique")

            return AIMessage("final")

    model = _EchoingModel()
    monkeypatch.setattr(debate_module, "get_chat_model", lambda temperature=0.1: model)

    await debate_graph.ainvoke(
        {"question": "q", "context": "c", "advocate_argument": "", "critic_argument": "", "final_answer": ""}
    )

    assert model.critic_saw is not None
    assert "Distinctive advocate claim #42." in model.critic_saw


async def test_moderator_receives_both_argument_and_critique(monkeypatch):
    class _EchoingModel:
        def __init__(self):
            self.moderator_saw = None

        async def ainvoke(self, messages):
            text = messages[0].content if messages else ""

            if "arguing FOR the most likely answer" in text:
                return AIMessage("ARGUMENT-MARKER")
            if "Critique the argument" in text:
                return AIMessage("CRITIQUE-MARKER")

            self.moderator_saw = text
            return AIMessage("final")

    model = _EchoingModel()
    monkeypatch.setattr(debate_module, "get_chat_model", lambda temperature=0.1: model)

    await debate_graph.ainvoke(
        {"question": "q", "context": "c", "advocate_argument": "", "critic_argument": "", "final_answer": ""}
    )

    assert "ARGUMENT-MARKER" in model.moderator_saw
    assert "CRITIQUE-MARKER" in model.moderator_saw
