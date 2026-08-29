from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from backend.app.context_engine.summarizer import ConversationSummarizer


def _messages(n: int) -> list[HumanMessage]:
    return [HumanMessage(f"message {i}") for i in range(n)]


async def test_short_conversation_is_not_summarized():
    summarizer = ConversationSummarizer(chat_model=GenericFakeChatModel(messages=iter([])))

    result = await summarizer.maybe_summarize(_messages(5))

    assert result.was_summarized is False
    assert result.kept_messages == _messages(5)


async def test_long_conversation_is_summarized_and_recent_messages_kept_verbatim():
    model = GenericFakeChatModel(messages=iter([AIMessage("Summary of the earlier discussion.")]))
    summarizer = ConversationSummarizer(chat_model=model)
    messages = _messages(15)

    result = await summarizer.maybe_summarize(messages)

    assert result.was_summarized is True
    assert result.summary == "Summary of the earlier discussion."
    assert len(result.kept_messages) == 6
    assert result.kept_messages == messages[-6:]


async def test_summarizer_does_not_touch_message_content_of_kept_messages():
    model = GenericFakeChatModel(messages=iter([AIMessage("summary")]))
    summarizer = ConversationSummarizer(chat_model=model)
    messages = _messages(20)

    result = await summarizer.maybe_summarize(messages)

    assert all(isinstance(m, HumanMessage) for m in result.kept_messages)
    assert result.kept_messages[-1].content == "message 19"
