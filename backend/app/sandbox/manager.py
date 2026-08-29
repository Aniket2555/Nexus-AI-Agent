import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

from backend.app.config import get_settings
from backend.app.sandbox.docker_sandbox import ensure_image, ensure_proxy, get_docker_client
from backend.app.sandbox.network_policy import resolve_network
from backend.app.sandbox.resource_limiter import default_limits, to_container_kwargs
from backend.app.sandbox.result_sanitizer import sanitize_output

SandboxLanguage = Literal["python", "node", "shell"]


class SandboxError(Exception):
    """Base for every error this module raises."""


class UnsupportedLanguageError(SandboxError):
    pass


class SandboxResult(TypedDict):
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    truncated: bool
    success: bool


@dataclass(frozen=True)
class _LanguageSpec:
    image_setting: str
    dockerfile: str

    def command(self, code: str) -> list[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class _PythonSpec(_LanguageSpec):
    def command(self, code: str) -> list[str]:
        return ["python3", "-c", code]


@dataclass(frozen=True)
class _NodeSpec(_LanguageSpec):
    def command(self, code: str) -> list[str]:
        return ["node", "-e", code]


@dataclass(frozen=True)
class _ShellSpec(_LanguageSpec):
    def command(self, code: str) -> list[str]:
        return ["sh", "-c", code]


_LANGUAGES: dict[str, _LanguageSpec] = {
    "python": _PythonSpec("sandbox_image_python", "Dockerfile.python"),
    "node": _NodeSpec("sandbox_image_node", "Dockerfile.node"),
    "shell": _ShellSpec("sandbox_image_shell", "Dockerfile.shell"),
}


def _run_sync(
    spec: _LanguageSpec,
    code: str,
    timeout_seconds: float,
    domains: tuple[str, ...] | None,
    *,
    on_start: Callable[[str | None], None] | None = None,
) -> SandboxResult:
    settings = get_settings()
    image = getattr(settings, spec.image_setting)
    ensure_image(image, spec.dockerfile)

    network = resolve_network(domains)
    if network.proxied:
        ensure_proxy(domains if domains is not None else settings.egress_allowed_domains)

    kwargs = to_container_kwargs(default_limits(timeout=timeout_seconds))
    kwargs["network_mode"] = network.network_mode
    kwargs["environment"] = network.environment

    client = get_docker_client()
    container = client.containers.create(image, spec.command(code), **kwargs)

    timed_out = False
    exit_code = None
    try:
        container.start()
        if on_start is not None:
            on_start(container.id)

        try:
            status = container.wait(timeout=timeout_seconds)
            exit_code = status.get("StatusCode")
        except Exception:
            timed_out = True
            try:
                container.kill()
            except Exception:
                pass

        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", "replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", "replace")

        if exit_code is None and not timed_out:
            container.reload()
            exit_code = container.attrs.get("State", {}).get("ExitCode")
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass
        if on_start is not None:
            on_start(None)

    sanitized = sanitize_output(stdout, stderr, max_bytes=settings.sandbox_max_output_bytes)

    return SandboxResult(
        stdout=sanitized.stdout,
        stderr=sanitized.stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        truncated=sanitized.truncated,
        success=(not timed_out) and (exit_code == 0),
    )


async def run(
    language: str,
    code: str,
    *,
    timeout_seconds: float | None = None,
    domains: tuple[str, ...] | None = None,
    on_start: Callable[[str | None], None] | None = None,
) -> SandboxResult:
    """Execute `code` in an isolated, resource-limited Docker container and
    return its captured (sanitized) output. §7.1's single entry point — every
    caller (the `python_repl`/`shell` MCP tools, the Code specialist) goes
    through this rather than touching `docker_sandbox.py` directly.

    Run via `asyncio.to_thread`: `docker-py` is a synchronous client (see
    pyproject.toml's note on why), and container creation, the config-hash
    comparison in `ensure_proxy`, and the blocking `container.wait()` call all
    belong on a worker thread, not the event loop.

    `on_start`, if given, is called with the container's id right after it
    starts (and with `None` once it's been removed) — `workflows/manager.py`'s
    `cancel()` is the reason this exists: killing a container from outside
    this function is the only way to actually interrupt a run that's blocked
    on the worker thread, since `Task.cancel()` can't reach into a thread
    running a blocking Docker call.
    """
    spec = _LANGUAGES.get(language)
    if spec is None:
        raise UnsupportedLanguageError(
            f"Unsupported sandbox language {language!r}. Supported: {sorted(_LANGUAGES)}"
        )

    settings = get_settings()
    effective_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else settings.sandbox_default_timeout_seconds
    )

    return await asyncio.to_thread(
        _run_sync, spec, code, effective_timeout, domains, on_start=on_start
    )
