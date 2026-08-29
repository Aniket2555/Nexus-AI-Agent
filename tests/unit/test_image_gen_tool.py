import httpx
import pytest

from backend.app.mcp.tools.custom.image_gen import ImageGenerationNotConfiguredError, generate_image


async def test_raises_clearly_when_not_configured(monkeypatch):
    from backend.app import config

    config.get_settings.cache_clear()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ImageGenerationNotConfiguredError):
        await generate_image("a red bicycle")

    config.get_settings.cache_clear()


async def test_sends_the_expected_request_and_parses_a_url_response(monkeypatch):
    from backend.app import config

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    config.get_settings.cache_clear()

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["body"] = request.read()
        return httpx.Response(200, json={"data": [{"url": "https://example.com/img.png"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = await generate_image("a red bicycle", http_client=client)
    await client.aclose()

    assert result == "https://example.com/img.png"
    assert captured["url"] == "https://api.openai.com/v1/images/generations"
    assert captured["auth"] == "Bearer test-key-123"
    assert b"a red bicycle" in captured["body"]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config.get_settings.cache_clear()


async def test_parses_a_base64_response_into_a_data_uri(monkeypatch):
    from backend.app import config

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    config.get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"b64_json": "QUJD"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = await generate_image("a red bicycle", http_client=client)
    await client.aclose()

    assert result == "data:image/png;base64,QUJD"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config.get_settings.cache_clear()


async def test_raises_on_a_non_200_response(monkeypatch):
    from backend.app import config

    monkeypatch.setenv("OPENAI_API_KEY", "bad-key")
    config.get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        await generate_image("a red bicycle", http_client=client)

    await client.aclose()

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config.get_settings.cache_clear()
