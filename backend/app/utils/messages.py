from langchain_core.messages import AnyMessage


def as_text(message: AnyMessage) -> str:
    """Flatten message content to text.

    Agent Chat UI can send content as a list of typed blocks (text, image_url) rather
    than a plain string. Passing that list to an embeddings/summarization call raises,
    so keep only the text parts. Shared by rag_chat.py (Phase 1) and the context
    engine's summarizer (Phase 3) — both need the same flattening.
    """
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
