from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.app.graphs.specialists import browser as browser_module


class _FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.calls = []

    async def ainvoke(self, tool_input):
        self.calls.append(tool_input)
        return self.result


class _FakeClient:
    def __init__(self, tools):
        self._tools = tools

    async def get_tools(self, server=None):
        return self._tools


async def _egress_allowlist(monkeypatch, domains):
    from backend.app import config

    monkeypatch.setenv("EGRESS_ALLOWED_DOMAINS", str(domains).replace("'", '"'))
    config.get_settings.cache_clear()


async def test_fetch_node_blocked_by_egress_allowlist_before_touching_mcp(monkeypatch):
    await _egress_allowlist(monkeypatch, ["example.com"])

    def explode(*args, **kwargs):
        raise AssertionError("get_client must not be called for a denied URL")

    monkeypatch.setattr(browser_module, "get_client", explode)

    result = await browser_module.fetch_node({"url": "https://evil.com", "task": "x"})

    assert result["error"] is not None
    assert "evil.com" in result["error"]

    from backend.app import config

    config.get_settings.cache_clear()


async def test_fetch_node_reports_unreachable_mcp_server_clearly(monkeypatch):
    await _egress_allowlist(monkeypatch, ["example.com"])

    def broken_client(**kwargs):
        raise ConnectionError("simulated: no Node.js / server unreachable")

    monkeypatch.setattr(browser_module, "get_client", broken_client)

    result = await browser_module.fetch_node({"url": "https://example.com", "task": "x"})

    assert "Could not reach" in result["error"]

    from backend.app import config

    config.get_settings.cache_clear()


async def test_fetch_node_reports_missing_expected_tools(monkeypatch):
    await _egress_allowlist(monkeypatch, ["example.com"])

    unrelated_tool = _FakeTool("some_other_tool", "n/a")
    monkeypatch.setattr(
        browser_module, "get_client", lambda **kwargs: _FakeClient([unrelated_tool])
    )

    result = await browser_module.fetch_node({"url": "https://example.com", "task": "x"})

    assert "did not expose" in result["error"]

    from backend.app import config

    config.get_settings.cache_clear()


async def test_fetch_node_succeeds_with_navigate_and_content_tools(monkeypatch):
    await _egress_allowlist(monkeypatch, ["example.com"])

    navigate = _FakeTool("browser_navigate", "ok")
    snapshot = _FakeTool("browser_snapshot", [{"type": "text", "text": "Page body text."}])
    monkeypatch.setattr(
        browser_module, "get_client", lambda **kwargs: _FakeClient([navigate, snapshot])
    )

    result = await browser_module.fetch_node({"url": "https://example.com/page", "task": "x"})

    assert result["error"] is None
    assert result["page_content"] == "Page body text."
    assert navigate.calls == [{"url": "https://example.com/page"}]

    from backend.app import config

    config.get_settings.cache_clear()


async def test_summarize_node_reports_the_fetch_error_instead_of_calling_the_model():
    result = await browser_module.summarize_node(
        {"url": "x", "task": "x", "page_content": "", "error": "denied"}
    )

    assert "Could not complete" in result["summary"]
    assert "denied" in result["summary"]


async def test_run_browser_returns_a_specialist_result(monkeypatch):
    await _egress_allowlist(monkeypatch, ["example.com"])

    navigate = _FakeTool("navigate", "ok")
    content = _FakeTool("get_content", [{"type": "text", "text": "some page content"}])
    monkeypatch.setattr(
        browser_module, "get_client", lambda **kwargs: _FakeClient([navigate, content])
    )

    model = GenericFakeChatModel(messages=iter([AIMessage("Summary of the page.")]))
    monkeypatch.setattr(browser_module, "get_chat_model", lambda: model)

    result = await browser_module.run_browser(
        "task-1", "https://example.com", "what does the page say?"
    )

    assert result["task_id"] == "task-1"
    assert result["specialist"] == "browser"
    assert result["success"] is True
    assert result["summary"] == "Summary of the page."

    from backend.app import config

    config.get_settings.cache_clear()
