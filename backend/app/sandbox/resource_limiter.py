from dataclasses import dataclass
from typing import Any

from backend.app.config import get_settings


@dataclass(frozen=True)
class ResourceLimits:
    """§7.1's resource-limit table, as one value object rather than five loose
    settings threaded through every call site.
    """

    cpu_cores: float
    memory_mb: int
    tmp_mb: int
    pids_limit: int
    timeout_seconds: float


def default_limits(*, timeout_seconds: float | None = None) -> ResourceLimits:
    settings = get_settings()
    return ResourceLimits(
        settings.sandbox_cpu_cores,
        settings.sandbox_memory_mb,
        settings.sandbox_tmp_mb,
        settings.sandbox_pids_limit,
        timeout_seconds=(
            timeout_seconds
            if timeout_seconds is not None
            else settings.sandbox_default_timeout_seconds
        ),
    )


def to_container_kwargs(limits: ResourceLimits) -> dict[str, Any]:
    """Translate `ResourceLimits` into `docker.DockerClient.containers.create()`
    kwargs enforcing §7.1's table:

    - CPU: `nano_cpus` (Docker's own CFS-quota knob, in billionths of a core).
    - Memory: `mem_limit` == `memswap_limit`, so the container can't spill into
      swap once it hits the memory ceiling — a bare `mem_limit` alone still
      allows swapping up to 2x by Docker's default, which would let a memory
      bomb degrade the host instead of getting OOM-killed at the configured
      limit.
    - Disk: no writable disk at all (`read_only=True`) except a size-capped
      `tmpfs` at `/tmp` — real files never touch the host filesystem or a
      container layer.
    - `pids_limit` and `cap_drop`/`security_opt` are hardening beyond the
      literal §7.1 table (which only lists CPU/memory/disk/timeout): a
      fork-bomb loop stays under the memory ceiling for a long time while
      exhausting the process table, and dropping all capabilities plus
      no-new-privileges costs nothing for code that has no legitimate need
      for any of them.
    """
    mem = f"{limits.memory_mb}m"

    return {
        "mem_limit": mem,
        "memswap_limit": mem,
        "nano_cpus": int(limits.cpu_cores * 1000000000),
        "pids_limit": limits.pids_limit,
        "read_only": True,
        "tmpfs": {"/tmp": f"size={limits.tmp_mb}m,mode=1777"},
        "working_dir": "/tmp",
        "user": "sandbox",
        "security_opt": ["no-new-privileges"],
        "cap_drop": ["ALL"],
        "network_disabled": False,
    }
