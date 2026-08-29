import re

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("OPENAI_KEY", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("JWT", re.compile(r"\bey[A-Za-z0-9_-]+\.ey[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")),
    (
        "PRIVATE_KEY_BLOCK",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]+?"
            r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
    ),
    (
        "GENERIC_CREDENTIAL",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*"
            r"['\"]?([A-Za-z0-9\-_/+=]{12,})['\"]?"
        ),
    ),
]


def redact_secrets(text: str) -> tuple[str, list[str]]:
    """Scrub known credential shapes from `text` (§5.5's "Secret redaction").

    Returns `(redacted_text, labels_found)` — the labels let a caller log *that*
    something was redacted (and of what kind) without logging the secret itself.
    """
    labels_found: list[str] = []

    def _make_sub(label: str):
        def _sub(match: re.Match) -> str:
            labels_found.append(label)
            return f"[REDACTED:{label}]"

        return _sub

    result = text
    for label, pattern in _PATTERNS:
        result = pattern.sub(_make_sub(label), result)
    return result, labels_found
