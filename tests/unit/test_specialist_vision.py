from backend.app.graphs.specialists import vision as vision_module


async def test_run_vision_reuses_the_phase_5_analyze_image_tool(monkeypatch):
    calls = []

    async def fake_analyze_image(image_url, question):
        calls.append((image_url, question))
        return "A red bicycle leaning against a wall."

    monkeypatch.setattr(vision_module, "analyze_image", fake_analyze_image)

    result = await vision_module.run_vision(
        "task-1", "https://example.com/bike.jpg", "What is in this image?"
    )

    assert result["task_id"] == "task-1"
    assert result["specialist"] == "vision"
    assert result["success"] is True
    assert result["summary"] == "A red bicycle leaning against a wall."
    assert calls == [("https://example.com/bike.jpg", "What is in this image?")]


async def test_run_vision_uses_the_default_question_when_none_given(monkeypatch):
    async def fake_analyze_image(image_url, question):
        assert question == vision_module.DEFAULT_QUESTION
        return "described"

    monkeypatch.setattr(vision_module, "analyze_image", fake_analyze_image)

    result = await vision_module.run_vision("task-1", "https://example.com/img.png")

    assert result["summary"] == "described"
