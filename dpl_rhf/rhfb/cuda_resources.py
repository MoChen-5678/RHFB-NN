"""CUDA allocation limits and runtime evidence; no silent CPU fallback.

The PyTorch caching allocator enforces its own budget. Driver/context and
third-party allocations are outside that allocator; the protected launcher
also monitors per-process driver memory when a GPU budget is requested.
"""
import csv
import json
import os
from pathlib import Path
import subprocess

from .execution_limits import ResourceSafetyError, require_bounded_cgroup

MAX_CUDA_BUDGET_MIB = 24576  # 2026-10-30: up to 24 GiB per device on GPU04
# 2026-10-15: physical GPU 1 only.  2026-10-27: the user authorized physical
# GPUs 0 and 1, so one job may span both (never any other index).
USER_AUTHORIZED_GPU_INDICES = frozenset({0, 1})


def gpu_inventory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], capture_output=True, text=True,
        check=True, timeout=5)
    return [{'index': int(index), 'uuid': uuid.strip(), 'name': name.strip(),
             'memory_total_mib': int(total), 'memory_used_mib': int(used),
             'utilization_percent': int(util)}
            for index, uuid, name, total, used, util in csv.reader(result.stdout.splitlines())]


def _authorized_indices(limits):
    """Local policy {0,1}; on a process-guarded remote host, the guard's list."""
    if limits.get('mode') == 'process_guard':
        return frozenset(int(i) for i in limits.get('authorized_gpu_indices') or ())
    return USER_AUTHORIZED_GPU_INDICES


def configure_cuda(device='cuda:0', budget_mib=8192):
    limits = require_bounded_cgroup()
    authorized = _authorized_indices(limits)
    import torch
    device = torch.device(device)
    if device.type != 'cuda':
        raise ResourceSafetyError('configure_cuda requires an explicit CUDA device')
    if isinstance(budget_mib, bool) or not 256 <= budget_mib <= MAX_CUDA_BUDGET_MIB:
        raise ResourceSafetyError(f'CUDA allocator budget must be 256..{MAX_CUDA_BUDGET_MIB} MiB')
    index = device.index if device.index is not None else 0
    # The launcher exposes exactly the selected UUIDs, in order, so local
    # ordinals have unambiguous physical identities. Refuse direct unbudgeted
    # CUDA launches.
    selected_uuid = os.environ.get('RHFB_CUDA_UUID')
    launcher_budget = os.environ.get('RHFB_CUDA_BUDGET_MIB')
    if not selected_uuid or launcher_budget is None:
        raise ResourceSafetyError('CUDA needs the launcher --gpu-uuid and --gpu-memory-mib policy')
    if int(launcher_budget) != budget_mib or os.environ.get('CUDA_VISIBLE_DEVICES') != selected_uuid:
        raise ResourceSafetyError('CUDA launcher identity/budget differs from the numerical job')
    uuids = selected_uuid.split(',')
    inventory = gpu_inventory()
    physical = []
    for uuid in uuids:
        item = next((gpu for gpu in inventory if gpu['uuid'] == uuid), None)
        if item is None:
            raise ResourceSafetyError('selected GPU UUID is no longer present')
        if item['index'] not in authorized:
            raise ResourceSafetyError(
                'user authorization permits only physical GPU indices '
                + ' and '.join(str(i) for i in sorted(authorized)))
        physical.append(item['index'])
    # Resolve host authorization before any operation that could initialize
    # a CUDA context on the selected device.
    if not torch.cuda.is_available():
        raise ResourceSafetyError('CUDA was requested but is unavailable; CPU fallback is refused')
    if index != 0 or torch.cuda.device_count() != len(uuids):
        raise ResourceSafetyError('the protected GPU job must see exactly its selected devices, primary as cuda:0')
    budget = budget_mib*1024**2
    context_headroom = min(512, budget_mib//4)*1024**2
    allocator_limit = budget-context_headroom
    totals = []
    for local in range(len(uuids)):
        total = int(torch.cuda.get_device_properties(local).total_memory)
        if budget >= total:
            raise ResourceSafetyError('CUDA budget must leave device headroom')
        totals.append(total)
    # The same allocator limit is enforced separately on every selected device.
    for local, total in enumerate(totals):
        torch.cuda.set_per_process_memory_fraction(allocator_limit/total, local)
        torch.cuda.reset_peak_memory_stats(local)
    torch.cuda.set_device(index)
    total = totals[0]
    evidence = {'device': str(torch.device('cuda', index)), 'uuid': selected_uuid,
            'physical_index': physical[0] if len(physical) == 1 else physical,
            'device_count': len(uuids),
            'name': torch.cuda.get_device_name(index), 'device_total_bytes': total,
            'allocator_limit_bytes': allocator_limit, 'allocator_limit_enforced': True,
            'driver_monitor_budget_bytes': budget, 'context_headroom_bytes': context_headroom,
            'driver_budget_guard_requested': True,
            'scope': 'PyTorch allocator hard budget; launcher monitors charged driver memory separately'}
    evidence_path = os.environ.get('RHFB_CUDA_POLICY_EVIDENCE')
    if not evidence_path:
        raise ResourceSafetyError('CUDA allocator policy needs the protected supervisor evidence path')
    path = Path(evidence_path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(evidence, indent=2)+'\n')
    temporary.replace(path)
    return evidence


def cuda_snapshot(device='cuda:0'):
    import torch
    device = torch.device(device)
    if device.type != 'cuda':
        return {'device': str(device), 'cuda_used': False}
    return {'device': str(device), 'uuid': os.environ.get('RHFB_CUDA_UUID'),
            'cuda_used': True,
            'allocated_bytes': torch.cuda.memory_allocated(device),
            'reserved_bytes': torch.cuda.memory_reserved(device),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
