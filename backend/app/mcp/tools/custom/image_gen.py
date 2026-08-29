from typing import Any

import httpx
from langchain_core.tools import tool

from backend.app.config import get_settings

OPENAI_IMAGES_URL = "https://api.openai.com/v1/images/generations"


class ImageGenerationNotConfiguredError(Exception):
    """No image generation provider is configured.

    Groq — this project's default chat provider (DECISIONS.md D7) — is
    chat/vision/audio only; it has no image generation endpoint, the same gap
    that ruled it out for embeddings in Phase 1. Unlike embeddings, there's no
    free local model that fills this one the way `sentence-transformers` did —
    image generation needs either a paid API or a locally-run diffusion model
    (real GPU/CPU cost either way). This module calls OpenAI's Images API when
    `OPENAI_API_KEY` is set (already an optional field in Settings for exactly
    this kind of provider-agnostic extensibility, DECISIONS.md D4); with no key,
    it raises this clearly rather than silently returning nothing.
    """


async def generate_image(
    prompt: str, size: str = "1024x1024", *, http_client: httpx.AsyncClient | None = None
) -> str:
    """Generate an image from a text prompt. Returns a URL or a `data:` URI,
    whichever OpenAI's response contains (varies by model/response_format).
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise ImageGenerationNotConfiguredError(
            "Image generation requires OPENAI_API_KEY to be set — no provider is "
            "configured by default (see this module's docstring)."
        )

    client = http_client or httpx.AsyncClient()
    try:
        response = await client.post(
            OPENAI_IMAGES_URL,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": settings.image_gen_model,
                "prompt": prompt,
                "size": size,
                "n": 1,
            },
            timeout=settings.tool_call_timeout_seconds,
        )
        response.raise_for_status()
        return _extract_image(response.json())
    finally:
        if http_client is None:
            await client.aclose()


def _extract_image(payload: dict[str, Any]) -> str:
    item = payload["data"][0]
    if "url" in item:
        return item["url"]
    if "b64_json" in item:
        return f"data:image/png;base64,{item['b64_json']}"
    raise ValueError(f"Unrecognized image generation response shape: {sorted(item)}")


@tool
async def generate_image_tool(prompt: str, size: str = "1024x1024") -> str:
    """Generate an image from a text prompt. Returns a URL or a data: URI. Raises
    if no image generation provider is configured (requires OPENAI_API_KEY).
    """
    return await generate_image(prompt, size=size)
