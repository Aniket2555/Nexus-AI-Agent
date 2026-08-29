import pytest

from backend.app.security.redaction import redact_secrets


@pytest.mark.parametrize(
    ("text", "expected_label"),
    [
        ("key: AKIAIOSFODNN7EXAMPLE", "AWS_ACCESS_KEY"),
        ("token=ghp_1234567890abcdefghijklmnopqrstuvwxyz12", "GITHUB_TOKEN"),
        ("xoxb-fake-0000000000-testfixturevalue", "SLACK_TOKEN"),
        ("sk-abcdefghijklmnopqrstuvwxyz1234567890ABCD", "OPENAI_KEY"),
        (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
            "JWT",
        ),
        ("password: hunter2superlongpassword123", "GENERIC_CREDENTIAL"),
    ],
)
def test_redacts_known_credential_shapes(text, expected_label):
    redacted, labels = redact_secrets(text)
    assert expected_label in labels
    assert f"[REDACTED:{expected_label}]" in redacted


def test_private_key_block_is_fully_redacted():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA1c\n-----END RSA PRIVATE KEY-----"
    redacted, labels = redact_secrets(text)
    assert labels == ["PRIVATE_KEY_BLOCK"]
    assert "MIIEpAIBAAKCAQEA1c" not in redacted


def test_ordinary_prose_is_left_untouched():
    text = "NEXUS uses Qdrant for dense retrieval and Groq for chat."
    redacted, labels = redact_secrets(text)
    assert redacted == text
    assert labels == []


def test_multiple_secrets_in_the_same_text_are_all_redacted():
    text = "aws key AKIAIOSFODNN7EXAMPLE and github token ghp_1234567890abcdefghijklmnopqrstuvwxyz12"
    redacted, labels = redact_secrets(text)
    assert set(labels) == {"AWS_ACCESS_KEY", "GITHUB_TOKEN"}
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "ghp_1234567890" not in redacted
