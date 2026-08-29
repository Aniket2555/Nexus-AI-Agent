from backend.app.sandbox.resource_limiter import (
    ResourceLimits,
    default_limits,
    to_container_kwargs,
)


def test_default_limits_reads_from_settings():
    limits = default_limits()

    assert limits.cpu_cores > 0
    assert limits.memory_mb > 0
    assert limits.timeout_seconds > 0


def test_default_limits_timeout_override_wins_over_settings():
    limits = default_limits(timeout_seconds=5.0)
    assert limits.timeout_seconds == 5.0


def test_to_container_kwargs_derives_memswap_from_memory_to_disable_swap():
    limits = ResourceLimits(
        cpu_cores=2.0, memory_mb=512, tmp_mb=1024, pids_limit=64, timeout_seconds=60.0
    )

    kwargs = to_container_kwargs(limits)

    assert kwargs["mem_limit"] == kwargs["memswap_limit"] == "512m"


def test_to_container_kwargs_converts_cpu_cores_to_nano_cpus():
    limits = ResourceLimits(
        cpu_cores=1.5, memory_mb=512, tmp_mb=1024, pids_limit=64, timeout_seconds=60.0
    )

    kwargs = to_container_kwargs(limits)

    assert kwargs["nano_cpus"] == 1500000000


def test_to_container_kwargs_makes_root_read_only_with_a_sized_tmp():
    limits = ResourceLimits(
        cpu_cores=2.0, memory_mb=512, tmp_mb=256, pids_limit=64, timeout_seconds=60.0
    )

    kwargs = to_container_kwargs(limits)

    assert kwargs["read_only"] is True
    assert kwargs["tmpfs"]["/tmp"] == "size=256m,mode=1777"


def test_to_container_kwargs_drops_all_capabilities():
    limits = ResourceLimits(
        cpu_cores=2.0, memory_mb=512, tmp_mb=1024, pids_limit=64, timeout_seconds=60.0
    )

    kwargs = to_container_kwargs(limits)

    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["pids_limit"] == 64
