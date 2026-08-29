from langchain_core.tools import tool

from backend.app.sandbox.manager import SandboxResult, run

_LANGUAGE = "shell"


async def run_shell(command: str, *, timeout_seconds: float | None = None) -> SandboxResult:
    """Execute a POSIX shell command in the §7.1 Docker sandbox (alpine + busybox
    ash — no coreutils beyond what the base image ships) and return its
    captured, sanitized output.
    """
    return await run(_LANGUAGE, command, timeout_seconds=timeout_seconds)


@tool
async def shell(command: str) -> str:
    """Run a shell command in an isolated, network-disabled, resource-limited
    sandbox (alpine/busybox — a minimal shell environment, not a full Linux
    distro) and return its stdout/stderr. No filesystem persists between calls
    and execution is killed after the configured timeout.
    """
    result = await run_shell(command)

    lines = [f"exit_code={result['exit_code']}", f"stdout:\n{result['stdout']}"]
    if result["stderr"]:
        lines.append(f"stderr:\n{result['stderr']}")
    if result["timed_out"]:
        lines.append("(execution timed out and was killed)")
    if result["truncated"]:
        lines.append("(output truncated)")

    return "\n".join(lines)
