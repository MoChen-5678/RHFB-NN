"""Pure launcher-bound checks; these tests never start a numerical payload."""
import importlib.util
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).resolve().parents[3] / "scripts" / "run_rhfb_limited.py"
SPEC = importlib.util.spec_from_file_location("rhfb_limited_launcher", LAUNCHER)
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


def test_default_mode_keeps_the_thirty_minute_maximum():
    GUARD.validate_runtime_arguments(4096, 1800, 8192)
    with pytest.raises(ValueError, match="--long-run"):
        GUARD.validate_runtime_arguments(4096, 1801, 8192)


def test_long_run_mode_has_a_finite_fourteen_day_maximum():
    maximum = 14 * 24 * 60 * 60
    GUARD.validate_runtime_arguments(4096, maximum, 8192, long_run=True)
    with pytest.raises(ValueError, match="14 days"):
        GUARD.validate_runtime_arguments(4096, maximum + 1, 8192, long_run=True)
    with pytest.raises(ValueError):
        GUARD.validate_runtime_arguments(4096, maximum, 8192)


@pytest.mark.parametrize("memory_mib", [31, 4097])
def test_long_run_never_relaxes_kernel_memory_cap(memory_mib):
    with pytest.raises(ValueError, match="MemoryMax"):
        GUARD.validate_runtime_arguments(memory_mib, 3600, 8192, long_run=True)


@pytest.mark.parametrize("gpu_mib", [255, 8193])
def test_long_run_never_relaxes_gpu_monitor_budget(gpu_mib):
    with pytest.raises(ValueError, match="GPU"):
        GUARD.validate_runtime_arguments(4096, 3600, gpu_mib, long_run=True)


def test_long_run_accepts_smallest_host_budget_without_changing_bounds():
    GUARD.validate_runtime_arguments(32, 3600, 256, long_run=True)
