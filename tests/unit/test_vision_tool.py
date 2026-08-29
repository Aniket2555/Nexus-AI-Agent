from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.app.mcp.tools.custom.vision import analyze_image


async def test_sends_image_and_question_as_a_multimodal_message():
    model = GenericFakeChatModel(messages=iter([AIMessage("A photo of a cat.")]))

    result = await analyze_image(
        "data:image/png;base64,AAA", "What is in this image?", chat_model=model
    )

    assert result == "A photo of a cat."


async def test_uses_the_default_question_when_none_given():
    model = GenericFakeChatModel(messages=iter([AIMessage("described")]))

    result = await analyze_image("https://example.com/photo.jpg", chat_model=model)

    assert result == "described"
