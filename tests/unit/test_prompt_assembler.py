from backend.app.context_engine.prompt_assembler import assemble_system_prompt


def test_base_sections_always_present():
    prompt = assemble_system_prompt("some context")
    assert "NEXUS" in prompt
    assert "some context" in prompt
    assert "Cite every claim" in prompt


def test_untrusted_content_instruction_always_present():
    """§5.5: the boundary instruction must precede the retrieved context on every
    render, not just when a caller remembers to ask for it."""
    prompt = assemble_system_prompt("<untrusted_content>ctx</untrusted_content>")
    assert "never instructions" in prompt.lower()

    instruction_pos = prompt.lower().index("never instructions")
    context_pos = prompt.index("<untrusted_content>ctx</untrusted_content>")
    assert instruction_pos < context_pos


def test_intent_guidance_section_appears_only_when_non_empty():
    code_prompt = assemble_system_prompt("ctx", intent="code")
    qa_prompt = assemble_system_prompt("ctx", intent="qa")

    assert "Guidance for this question type" in code_prompt
    assert "runnable code blocks" in code_prompt
    assert "Guidance for this question type" not in qa_prompt


def test_memory_summary_section_is_a_toggle():
    with_summary = assemble_system_prompt("ctx", memory_summary="Earlier: discussed PTO.")
    without_summary = assemble_system_prompt("ctx", memory_summary=None)

    assert "Summary of earlier conversation" in with_summary
    assert "discussed PTO" in with_summary
    assert "Summary of earlier conversation" not in without_summary


def test_unknown_intent_falls_back_to_no_guidance_section():
    prompt = assemble_system_prompt("ctx", intent="not-a-real-intent")
    assert "Guidance for this question type" not in prompt
