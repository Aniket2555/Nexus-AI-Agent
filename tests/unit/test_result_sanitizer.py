from backend.app.sandbox.result_sanitizer import redact_secrets, sanitize_output


def test_redacts_openai_style_key():
    text = "here is my key sk-abcdefghijklmnopqrstuvwx, don't share it"
    assert "sk-abcdefghijklmnopqrstuvwx" not in redact_secrets(text)
    assert "[REDACTED]" in redact_secrets(text)


def test_redacts_groq_style_key():
    assert "[REDACTED]" in redact_secrets("token: gsk_abcdefghijklmnopqrstuvwx1234")


def test_redacts_aws_access_key_id():
    assert "[REDACTED]" in redact_secrets("AKIA1234567890ABCDEF is the id")


def test_redacts_labelled_assignment():
    text = 'PASSWORD="hunter2plus"'
    assert "hunter2plus" not in redact_secrets(text)


def test_leaves_ordinary_text_untouched():
    text = "The answer is 42. No secrets here."
    assert redact_secrets(text) == text


def test_sanitize_output_caps_each_stream_independently():
    stdout = "a" * 100
    stderr = "b" * 10

    result = sanitize_output(stdout, stderr, max_bytes=20)

    assert len(result.stdout.encode("utf-8")) <= 20 + len("\n...[truncated]")
    assert result.stderr == "b" * 10
    assert result.truncated is True


def test_sanitize_output_not_truncated_when_within_budget():
    result = sanitize_output("short", "also short", max_bytes=1000)

    assert result.truncated is False
    assert result.stdout == "short"
    assert result.stderr == "also short"


def test_sanitize_output_redacts_before_capping():
    long_prefix = "x" * 50
    text = f"{long_prefix} sk-abcdefghijklmnopqrstuvwx"

    result = sanitize_output(text, "", max_bytes=1000)

    assert "sk-abcdefghijklmnopqrstuvwx" not in result.stdout
