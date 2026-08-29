from langchain_core.tools import tool

from backend.app.sandbox.manager import SandboxResult, run

_LANGUAGE = "python"


async def run_python(code: str, *, timeout_seconds: float | None = None) -> SandboxResult:
    """Execute Python source in the §7.1 Docker sandbox and return its captured,
    sanitized output. Deferred here from Phase 5 (§7.1's note) — this is the
    first thing in the project that actually *runs* arbitrary generated code
    rather than only validating it (Phase 6's Code specialist did `ast.parse()`
    only, for exactly this reason).
    """
    return await run(_LANGUAGE, code, timeout_seconds=timeout_seconds)


@tool
async def python_repl(code: str) -> str:
    """Execute Python code in an isolated, network-disabled, resource-limited
    sandbox and return its stdout/stderr. No filesystem persists between calls,
    no packages beyond the standard library are available, and execution is
    killed after the configured timeout. Use for calculations, data
    transformations, or verifying that generated code actually runs — not for
    anything that needs network access or installed dependencies.
    """
    result = await run_python(code)

    lines = [f"exit_code={result['exit_code']}", f"stdout:\n{result['stdout']}"]
    if result["stderr"]:
        lines.append(f"stderr:\n{result['stderr']}")
    if result["timed_out"]:
        lines.append("(execution timed out and was killed)")
    if result["truncated"]:
        lines.append("(output truncated)")

    return "\n".join(lines)
