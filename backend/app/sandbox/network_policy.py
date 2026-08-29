from dataclasses import dataclass, field
from pathlib import Path

from backend.app.config import get_settings

_SQUID_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[3] / "infra" / "docker" / "sandbox" / "squid.conf.template"
)


@dataclass(frozen=True)
class NetworkConfig:
    """What `docker_sandbox.py` needs to attach a container to the right network
    and (if applicable) point it at the egress proxy.
    """

    network_mode: str
    environment: dict[str, str] = field(default_factory=dict)
    proxied: bool = False


def resolve_network(domains: tuple[str, ...] | None = None) -> NetworkConfig:
    """Deny-all by default (§7.1's default posture, matching `egress_allowed_domains`'s
    own fail-closed default from Phase 5): with no domains configured,
    `network_mode="none"` gives the container no network namespace at all — not
    even loopback to the host — which is a stronger guarantee than any allowlist
    could be and needs no proxy running.

    With domains configured, sandbox containers can't be trusted to make their
    own allowlist decision the way `security/egress.py`'s `check_url_allowed`
    does for a structured tool call (`url=...` argument) — code running inside
    the sandbox is opaque, it can construct or obfuscate a URL however it wants.
    So enforcement moves down a layer, to the network itself: the container
    joins an *internal* Docker network (no route to the internet at all) whose
    only reachable peer is the squid proxy, which is the thing that actually
    enforces the allowlist, at the TCP/TLS-SNI level, via `dstdomain` ACLs. The
    setting itself (`egress_allowed_domains`) is reused as-is from Phase 5 — only
    the enforcement mechanism differs, because the input differs (arbitrary code
    vs. one structured argument).
    """
    settings = get_settings()
    allowed = domains if domains is not None else settings.egress_allowed_domains
    if not allowed:
        return NetworkConfig(network_mode="none")

    proxy_url = f"http://{settings.sandbox_proxy_host}:{settings.sandbox_proxy_port}"
    return NetworkConfig(
        settings.sandbox_egress_network,
        environment={
            "HTTP_PROXY": proxy_url,
            "HTTPS_PROXY": proxy_url,
            "http_proxy": proxy_url,
            "https_proxy": proxy_url,
            "NO_PROXY": "",
            "no_proxy": "",
        },
        proxied=True,
    )


def render_squid_config(domains: tuple[str, ...] | None) -> str:
    """Fill `squid.conf.template`'s `__ALLOWED_DOMAINS__` placeholder with a
    squid `dstdomain` ACL list.

    Bare domains only (`example.com`, not also `.example.com`): squid's
    `dstdomain` already matches a bare domain's subdomains (`api.example.com`
    matches an `example.com` entry) — verified directly against a live squid
    5.7 container, which refuses to start on `dstdomain example.com
    .example.com` with `ERROR: '.example.com' is a subdomain of 'example.com'
    / FATAL: Bungled /etc/squid/squid.conf`. Adding the `.`-prefixed form
    (which squid's own docs describe as "matches the domain and subdomains",
    reading like it should be necessary) is what actually broke this — it's
    already redundant with the bare form.
    """
    template = _SQUID_TEMPLATE_PATH.read_text(encoding="utf-8")
    acl_domains = " ".join(domains) if domains else "invalid.example"
    return template.replace("__ALLOWED_DOMAINS__", acl_domains)
