import os
import re
from pathlib import Path
from typing import Any

import yaml
from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVERS_PATH = Path(__file__).parent / "servers.yaml"
_VAR_PATTERN = re.compile(r"\$\{(\w+)(:-(.*?))?\}")


def _substitute_env(value: Any) -> Any:
    """Resolve `${VAR}` / `${VAR:-default}` references from the environment.

    A bare `${VAR}` with no default and no env value resolves to `""` — deliberately
    not an error here, since the whole point of `requires_env` (below) is to catch
    "this server needs a token it doesn't have" *before* a blank credential gets
    used, not by making substitution itself picky about every string field.
    """
    if isinstance(value, str):

        def replace(match: re.Match) -> str:
            var, _, default = match.group(1), match.group(2), match.group(3)
            return os.environ.get(var, default if default is not None else "")

        return _VAR_PATTERN.sub(replace, value)

    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_substitute_env(v) for v in value]

    return value


def load_server_configs(path: Path = _SERVERS_PATH) -> dict[str, dict[str, Any]]:
    """Load `servers.yaml`, dropping any server whose `requires_env` variables
    aren't all set in the environment — e.g. `github` needs `GITHUB_TOKEN`; without
    it, the server is silently omitted rather than offered with a blank bearer
    token that would fail loudly (and confusingly) on first real call.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    configs: dict[str, dict[str, Any]] = {}

    for name, spec in raw.items():
        spec = dict(spec)
        required_vars = spec.pop("requires_env", [])
        if any(not os.environ.get(var) for var in required_vars):
            continue
        configs[name] = _substitute_env(spec)

    return configs


def get_client(
    server_names: list[str] | None = None, extra_configs: dict[str, dict[str, Any]] | None = None
) -> MultiServerMCPClient:
    """Build a client over the configured (and available) servers.

    `extra_configs` lets a caller — a test, or a specialist agent with its own
    tool needs — add server configs beyond `servers.yaml` without editing the
    shared registry.
    """
    configs = load_server_configs()
    if extra_configs:
        configs.update(extra_configs)

    if server_names is not None:
        configs = {name: cfg for name, cfg in configs.items() if name in server_names}

    return MultiServerMCPClient(configs)
