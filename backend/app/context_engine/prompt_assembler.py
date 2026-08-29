from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from backend.app.security.content_boundary import UNTRUSTED_CONTENT_INSTRUCTION

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)

INTENT_GUIDANCE: dict[str, str] = {
    "code": (
        "Prefer precise, runnable code blocks over prose. State any assumptions "
        "about language, version, or framework explicitly rather than guessing "
        "silently."
    ),
    "creative": (
        "Creative requests may draw on general knowledge beyond the retrieved "
        "context, but never present un-sourced content as if it were cited from a "
        "document."
    ),
    "analysis": (
        "State your reasoning steps before the conclusion. If the retrieved "
        "context is insufficient for a confident comparison, say so explicitly "
        "rather than guessing."
    ),
    "search": "Prioritize directly quoting the matching passage over paraphrasing it.",
    "qa": "",
}


def assemble_system_prompt(
    context: str, *, intent: str = "qa", memory_summary: str | None = None
) -> str:
    """Render the RAG chat system prompt from the Jinja2 template, with the
    intent-specific guidance and conversation-summary sections included only when
    there's something to put in them.
    """
    template = _env.get_template("agents/rag_chat.jinja2")
    return template.render(
        context=context,
        intent_instructions=INTENT_GUIDANCE.get(intent, ""),
        memory_summary=memory_summary,
        untrusted_content_instruction=UNTRUSTED_CONTENT_INSTRUCTION,
    )
