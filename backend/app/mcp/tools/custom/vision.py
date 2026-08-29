from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from backend.app.config import get_settings
from backend.app.llm.provider import get_chat_model
from backend.app.utils.messages import as_text

DEFAULT_QUESTION = "Describe this image in detail."


async def analyze_image(
    image_url: str,
    question: str = DEFAULT_QUESTION,
    *,
    chat_model: BaseChatModel | None = None,
) -> str:
    """Analyze an image with a vision-capable chat model (§5.2).

    `image_url` follows the same multimodal content-block convention Agent Chat UI
    already sends for user-uploaded images (Phase 1's `_as_text`/`as_text` flattens
    the same shape on the way in) — a `data:` URI or an http(s) URL.

    Not resolved to `get_chat_model()` in a default argument — that would
    eagerly construct a real provider client the same way Phase 1/3/4's
    `OpenAIEmbeddings`/`ChatGroq`/`RedisCache`/`Neo4jGraph` did before being fixed.
    Resolved inside the function body instead, at call time.
    """
    settings = get_settings()
    model = chat_model or get_chat_model(settings.vision_model, temperature=0)

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )

    response = await model.ainvoke([message])
    return as_text(response)


@tool
async def analyze_image_tool(image_url: str, question: str = DEFAULT_QUESTION) -> str:
    """Analyze an image and answer a question about it (or describe it, by
    default). `image_url` may be a data: URI or an http(s) URL.
    """
    return await analyze_image(image_url, question)
