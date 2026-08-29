"""Exercises the real §7.1 Docker sandbox — real containers, real resource
limits, a real squid egress proxy. Requires a running Docker daemon reachable
via `docker.from_env()` (verified directly against Docker Desktop on this
project's dev machine) and builds the sandbox images/proxy image on first use
if they aren't already present (`docker_sandbox.ensure_image`).
"""

from backend.app.sandbox import manager

PROXY_DOMAIN = "example.com"


async def test_python_executes_for_real_and_captures_stdout_and_stderr():
    result = await manager.run(
        "python", "import sys; print('out'); print('err', file=sys.stderr)"
    )

    assert result["stdout"] == "out\n"
    assert result["stderr"] == "err\n"
    assert result["exit_code"] == 0
    assert result["success"] is True


async def test_node_executes_for_real():
    result = await manager.run("node", "console.log(1 + 2)")

    assert result["stdout"] == "3\n"
    assert result["success"] is True


async def test_shell_executes_for_real():
    result = await manager.run("shell", "whoami")

    assert result["stdout"] == "sandbox\n"


async def test_root_filesystem_is_read_only():
    result = await manager.run("python", "open('/etc/should_not_write', 'w').write('x')")

    assert result["success"] is False
    assert "Read-only file system" in result["stderr"]


async def test_tmp_is_writable():
    result = await manager.run(
        "python", "open('/tmp/ok.txt', 'w').write('x'); print(open('/tmp/ok.txt').read())"
    )

    assert result["stdout"] == "x\n"
    assert result["success"] is True


async def test_timeout_kills_a_runaway_process():
    result = await manager.run("python", "import time; time.sleep(30)", timeout_seconds=2)

    assert result["timed_out"] is True
    assert result["success"] is False


async def test_memory_limit_oom_kills_an_oversized_allocation():
    result = await manager.run(
        "python", "b = bytearray(700 * 1024 * 1024)", timeout_seconds=15
    )

    assert result["timed_out"] is False
    assert result["exit_code"] == 137
    assert result["success"] is False


async def test_secrets_printed_by_generated_code_are_redacted():
    code = "print('API_KEY =', 'sk-abcdefghijklmnopqrstuvwx')"
    result = await manager.run("python", code)

    assert "sk-abcdefghijklmnopqrstuvwx" not in result["stdout"]
    assert "[REDACTED]" in result["stdout"]


async def test_network_is_fully_disabled_by_default():
    code = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://example.com', timeout=5)\n"
        "    print('reached')\n"
        "except Exception as e:\n"
        "    print('blocked:', type(e).__name__)\n"
    )
    result = await manager.run("python", code, timeout_seconds=10)

    assert "blocked" in result["stdout"]
    assert "reached" not in result["stdout"]


async def test_allowlisted_domain_is_reachable_through_the_egress_proxy():
    code = (
        "import urllib.request\n"
        "print(urllib.request.urlopen('http://example.com', timeout=8).status)"
    )
    result = await manager.run("python", code, timeout_seconds=15, domains=(PROXY_DOMAIN,))

    assert result["stdout"].strip() == "200"
    assert result["success"] is True


async def test_non_allowlisted_domain_is_rejected_by_the_egress_proxy():
    code = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://neverssl.com', timeout=8)\n"
        "    print('reached')\n"
        "except Exception as e:\n"
        "    print('blocked:', type(e).__name__, e)\n"
    )
    result = await manager.run("python", code, timeout_seconds=15, domains=(PROXY_DOMAIN,))

    assert "blocked" in result["stdout"]
    assert "reached" not in result["stdout"]
