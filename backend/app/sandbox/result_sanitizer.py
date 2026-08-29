import re
from typing import NamedTuple

_SECRET_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"gsk_[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|authorization)\b\s*"
        r"[:=]\s*['\"]?[^\s'\"]{4,}['\"]?"
    ),
)

_REDACTED = "[REDACTED]"


def redact_secrets(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


class SanitizedOutput(NamedTuple):
    stdout: str
    stderr: str
    truncated: bool


def _cap(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= max_bytes:
        return text, False
    return encoded[:max_bytes].decode("utf-8", "ignore") + "\n...[truncated]", True


def sanitize_output(stdout: str, stderr: str, max_bytes: int) -> SanitizedOutput:
    """Redact secrets first, then cap size — redacting after truncation risks
    cutting a secret in half and leaving the surviving half unredacted.
    """
    stdout = redact_secrets(stdout)
    stderr = redact_secrets(stderr)

    stdout, out_truncated = _cap(stdout, max_bytes)
    stderr, err_truncated = _cap(stderr, max_bytes)

    return SanitizedOutput(stdout, stderr, truncated=out_truncated or err_truncated)
