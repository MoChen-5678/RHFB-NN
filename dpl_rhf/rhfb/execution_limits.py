"""Fail-closed resource checks; no Torch import or scientific allocation."""
import json
import os
from pathlib import Path, PurePosixPath

MAX_JOB_MEMORY_BYTES = 4 * 1024**3
# Remote hosts without a delegable cgroup v2 memory controller (2026-10-27 user
# decision): scripts/run_rhfb_procguard.py samples the job's resident memory and
# terminates it above this bound.  Local runs keep the kernel cgroup limit.
MAX_PROCESS_GUARD_BYTES = 8 * 1024**3
MAX_DENSE_COORDINATE_POINTS = 192
MAX_DENSE_COORDINATE_DIMENSION = 96


class ResourceSafetyError(RuntimeError):
    pass


def process_guard_evidence():
    """Validate a live process-level guard for this process, or return None."""
    path = os.environ.get('RHFB_PROCESS_GUARD')
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    if data.get('mode') != 'process_guard':
        raise ValueError('process guard evidence has the wrong mode')
    if os.getsid(0) != data.get('payload_sid'):
        raise ValueError('this process is not inside the guarded session')
    os.kill(int(data['guard_pid']), 0)  # raises if the supervisor is gone
    limit = int(data['memory_max_bytes'])
    if not 0 < limit <= MAX_PROCESS_GUARD_BYTES:
        raise ValueError('process guard memory limit is unset or above 8 GiB')
    return {'mode': 'process_guard', 'guard_evidence': path, 'memory_max_bytes': limit,
            'memory_swap_max_bytes': None, 'enforced': True,
            'authorized_gpu_indices': data.get('authorized_gpu_indices'),
            'scope': 'sampled resident memory of the guarded session; not a kernel limit'}


def require_bounded_cgroup(*, proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    """Require a real kernel memory.max <=4 GiB and zero allowed swap.

    Environment variables and post-run RSS reports are not enforcement.
    Refuse an unsupported/unlimited cgroup instead of silently falling back.
    """
    try:
        records = (Path(proc_root)/'self/cgroup').read_text().splitlines()
        relative = next(line[3:] for line in records if line.startswith('0::'))
        parts = PurePosixPath(relative).parts
        if '..' in parts:
            raise ValueError('invalid cgroup path')
        directory = Path(cgroup_root)/relative.lstrip('/')
        maximum = (directory/'memory.max').read_text().strip()
        swap = (directory/'memory.swap.max').read_text().strip()
        if maximum == 'max' or not 0 < int(maximum) <= MAX_JOB_MEMORY_BYTES:
            raise ValueError('memory.max is unlimited or exceeds 4 GiB')
        if swap != '0':
            raise ValueError('memory.swap.max must be zero')
    except (OSError, ValueError, StopIteration) as exc:
        try:
            guard = process_guard_evidence()
        except (OSError, ValueError, KeyError, TypeError) as guard_exc:
            raise ResourceSafetyError(
                'RHFB process-guard evidence is invalid: '+str(guard_exc)) from exc
        if guard is not None:
            return guard
        raise ResourceSafetyError(
            'RHFB numerical execution requires an enforced cgroup memory limit '
            '(<=4 GiB, swap=0). Use scripts/run_rhfb_limited.py. '+str(exc)) from exc
    return {'cgroup': str(directory), 'memory_max_bytes': int(maximum),
            'memory_swap_max_bytes': 0, 'enforced': True}


def require_small_coordinate_space(basis):
    """Quarantine the dense coordinate prototype before any R allocation.

    These are engineering admission limits, not physical space truncations.
    A rejected calculation is not rerun on a smaller physical space.
    """
    if (basis.point_count > MAX_DENSE_COORDINATE_POINTS
            or basis.dimension > MAX_DENSE_COORDINATE_DIMENSION):
        raise ResourceSafetyError(
            f'Dense coordinate prototype is quarantined for P={basis.point_count}, '
            f'D={basis.dimension}: admission requires P<={MAX_DENSE_COORDINATE_POINTS} '
            f'and D<={MAX_DENSE_COORDINATE_DIMENSION}. '
            'Full-size runs must await a bounded-memory implementation; '
            'do not substitute a smaller nucleus/grid or override this check.')
