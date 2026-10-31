"""Reports must distinguish a completed payload from missing execution evidence."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'rhfb_runtime_guard', Path(__file__).resolve().parents[3]/'scripts/run_rhfb_limited.py')
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


def report():
    limits = {'enforced': True, 'memory_max_bytes': 64*1024**2, 'memory_swap_max_bytes': 0}
    resources = {'memory.current': 123456, 'memory.peak': None,
                 'memory.events': dict(low=0, high=0, max=0, oom=0, oom_kill=0)}
    final = {**limits, 'worker_finalized': True, 'child_actual_exit_code': 0,
             'termination_reason': 'child_exited', 'descendants_clear': True,
             'resource_start': copy.deepcopy(resources), 'resource_final': resources,
             'kernel_limits_final': limits.copy()}
    return {'kernel_limit_evidence': final, 'memory_max_bytes': 64*1024**2,
            'exit_code': 0, 'systemd_result': 'LoadState=not-found\nResult=success'}


def test_older_kernel_without_memory_peak_can_prove_completion():
    value = report()
    assert GUARD.execution_status(value) == 'completed'
    assert value['completion_evidence_verified'] is True


@pytest.mark.parametrize('key', ['worker_finalized', 'child_actual_exit_code',
                                 'resource_final', 'kernel_limits_final', 'descendants_clear'])
def test_missing_terminal_evidence_is_incomplete_even_with_launcher_success(key):
    value = report()
    del value['kernel_limit_evidence'][key]
    assert GUARD.execution_status(value) == 'incomplete'


def test_oom_event_increment_is_an_oom_failure():
    value = report()
    value['kernel_limit_evidence']['resource_final']['memory.events']['oom_kill'] = 1
    assert GUARD.execution_status(value) == 'oom_killed'
    assert value['completion_evidence_verified'] is False


def test_absent_resource_manager_evidence_is_incomplete():
    value = report()
    value['systemd_result'] = ''
    assert GUARD.execution_status(value) == 'incomplete'


def test_gpu_monitor_only_cannot_prove_allocator_enforcement():
    value = report()
    value.update(gpu_uuid='example-device', gpu_driver_monitor_budget_mib=2048)
    assert GUARD.execution_status(value) == 'incomplete'
    assert value['cuda_allocator_evidence_missing'] is True
