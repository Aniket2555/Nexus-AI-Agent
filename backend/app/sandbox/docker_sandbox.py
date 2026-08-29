import hashlib
import io
import tarfile
from pathlib import Path

import docker
from docker.errors import ImageNotFound, NotFound

from backend.app.config import get_settings
from backend.app.sandbox.network_policy import render_squid_config

_SANDBOX_DIR = Path(__file__).resolve().parents[3] / "infra" / "docker" / "sandbox"

_client: "docker.DockerClient | None" = None


def get_docker_client() -> "docker.DockerClient":
    """Module-level lazy singleton — same reasoning as `dense.py`'s
    `DenseRetriever` (Phase 1) and every other per-process client in this
    codebase: constructing `docker.from_env()` per call would reopen the
    platform's Docker socket/named-pipe connection on every single sandbox run.
    """
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def ensure_image(tag: str, dockerfile: str) -> None:
    """Build `tag` from `infra/docker/sandbox/{dockerfile}` if it isn't already
    present locally. Phase 7 has no image registry/distribution story — build is
    the only path, and it's a one-time cost per (tag, Dockerfile) pair since
    Docker's own build cache makes repeat builds near-instant once the base
    layer is pulled.
    """
    client = get_docker_client()
    try:
        client.images.get(tag)
        return
    except ImageNotFound:
        pass

    client.images.build(path=str(_SANDBOX_DIR), dockerfile=dockerfile, tag=tag, rm=True)


def ensure_network(name: str) -> None:
    """The internal (`internal=True`) network sandbox containers and the proxy
    share. `internal=True` is what actually makes the network isolation real —
    without it, a user-defined bridge network still NATs out to the internet by
    default, which would make the whole allowlist pointless.
    """
    client = get_docker_client()
    try:
        client.networks.get(name)
        return
    except NotFound:
        pass

    client.networks.create(name, driver="bridge", internal=True)


def _tar_of(name: str, data: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def ensure_proxy(domains: tuple[str, ...]) -> None:
    """Start (or restart, on config change) the squid egress proxy.

    The rendered config is injected via `put_archive` rather than a bind mount:
    a bind mount needs a host path Docker can see, which on Windows means
    relying on Docker Desktop's path translation for whatever directory the
    config happens to be written to. `put_archive` (a tar stream over the
    Docker Engine API) works identically regardless of host OS or where the
    caller's temp file landed, so it's the more portable choice, not a
    Windows-only workaround.

    Config-change detection is a label holding a hash of the rendered config,
    not a fixed "does a container with this name exist" check — otherwise
    changing `egress_allowed_domains` and restarting the app would keep serving
    a proxy built from the old allowlist.
    """
    settings = get_settings()
    client = get_docker_client()

    ensure_network(settings.sandbox_egress_network)
    ensure_image(settings.sandbox_image_proxy, "Dockerfile.proxy")

    config_text = render_squid_config(domains)
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()[:16]

    try:
        existing = client.containers.get(settings.sandbox_proxy_host)
        current_hash = existing.labels.get("nexus.squid_config_hash")
        if current_hash == config_hash and existing.status == "running":
            return
        try:
            existing.remove(force=True)
        except NotFound:
            pass
    except NotFound:
        pass

    container = client.containers.create(
        settings.sandbox_image_proxy,
        name=settings.sandbox_proxy_host,
        network=settings.sandbox_egress_network,
        labels={"nexus.squid_config_hash": config_hash},
        detach=True,
    )
    container.put_archive("/etc/squid", _tar_of("squid.conf", config_text.encode()))
    container.start()

    client.networks.get("bridge").connect(container)
